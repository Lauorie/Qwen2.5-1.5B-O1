import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    max_length: int = 3000
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    log_iter: int = 400
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 1000
    num_epochs: int = 1
    save_steps: int = 2000
    model_name: str = "Qwen2.5-1.5B"
    model_path: str = "/root/app/models/Qwen2.5-1.5B"
    data_path: str = "/root/app/Reason/data/reasoning-base-20k"
    output_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_output_dir()
        self.setup_model_and_tokenizer()
        self.setup_training_components()
        self.initialize_logging()

    def setup_output_dir(self):
        """Create output directory if it doesn't exist"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        try:
            # Get available GPU memory
            if torch.cuda.is_available():
                gpu_memory = {i: torch.cuda.get_device_properties(i).total_memory 
                            for i in range(torch.cuda.device_count())}
                memory_config = {i: f"{gpu_memory[i] // (1024 ** 3)}GB" 
                               for i in range(len(gpu_memory))}
            else:
                memory_config = None

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype="auto",
                device_map="balanced" if torch.cuda.is_available() else None,
                max_memory=memory_config
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # Configure model generation settings
            self.configure_model_generation()
            
            # Add special token
            self.add_special_tokens()
            
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise

    def configure_model_generation(self):
        """Configure model generation parameters"""
        self.model.generation_config.do_sample = True
        self.model.generation_config.eos_token_id = [151645, 151643]
        self.model.generation_config.pad_token_id = 151643
        self.model.generation_config.temperature = 0.7
        self.model.generation_config.top_p = 0.8
        self.model.generation_config.top_k = 20
        self.model.generation_config.transformers_version = "4.45.2"
        self.model.generation_config.repetition_penalty = 1.05

    def add_special_tokens(self):
        """Add special tokens to tokenizer and resize model embeddings"""
        new_special_token = "<|reasoning|>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [new_special_token]}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def setup_training_components(self):
        """Initialize optimizer and other training components"""
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.max_lr
        )
        
        self.trainable_params = sum(
            p.numel() for p in filter(lambda p: p.requires_grad, self.model.parameters())
        )

    def prepare_dataset(self):
        """Load and prepare the dataset"""
        dataset = datasets.load_dataset(self.config.data_path)
        dataset['train'] = dataset['train'].map(
            self.prepare_conversation_template,
            remove_columns=dataset['train'].column_names
        )
        return dataset

    @staticmethod
    def prepare_conversation_template(example):
        """Prepare conversation template for each example"""
        return {
            'template_new': "".join([
                "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n",
                "<|im_start|>user\n", example['user'], "<|im_end|>\n",
                "<|im_start|><|reasoning|>\n", example['reasoning'], "<|im_end|>\n",
                "<|im_start|>assistant\n", example['assistant'], "<|im_end|>\n"
            ])
        }

    def prepare_batch(self, dataset, step):
        """Prepare a batch of data for training"""
        start_idx = step * self.config.batch_size
        end_idx = start_idx + self.config.batch_size
        
        # Get batch of input ids
        batch_inputids = dataset['train'][start_idx:end_idx]['template_new']
        
        # Tokenize and pad the batch
        batch_encoding = self.tokenizer(
            batch_inputids,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = batch_encoding['input_ids'].to(self.config.device)
        
        # Prepare inputs and targets for next-token prediction
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        
        # Create masks
        padding_mask = torch.where(y == self.model.generation_config.pad_token_id, 0, 1)
        answer_mask = self.create_answer_mask(x)
        combined_mask = (answer_mask & padding_mask)
        
        # Check if the batch contains valid answer tokens
        if combined_mask.sum(dim=-1).min().item() == 0:
            raise ValueError("Batch contains no valid answer tokens")
        
        return {
            'x': x,
            'y': y,
            'combined_mask': combined_mask
        }

    def create_answer_mask(self, input_ids):
        """Create mask for answer tokens"""
        assistant_answer_mask = torch.zeros_like(input_ids)
        
        for i in range(input_ids.shape[0]):
            # Find positions of im_end tokens
            im_end_token = self.tokenizer.encode('<|im_end|>')[0]
            im_end_positions = torch.where(input_ids[i] == im_end_token)[0]
            
            # Get user and assistant end positions
            user_end_positions = [pos + 1 for pos in im_end_positions[1::3]]
            assistant_end_positions = [pos + 1 for pos in im_end_positions[3::3]]
            
            # Handle different cases of conversation turns
            if len(user_end_positions) == len(assistant_end_positions):
                # Complete conversation turns
                for user_end, assistant_end in zip(user_end_positions, assistant_end_positions):
                    assistant_answer_mask[i][user_end+3:assistant_end-1] = 1
            elif len(user_end_positions) == len(assistant_end_positions) + 1:
                # Handle incomplete last turn
                if len(user_end_positions) == 1:
                    # Single turn case
                    assistant_answer_mask[i][user_end_positions[0]+3:] = 1
                else:
                    # Multi-turn case
                    for user_end, assistant_end in zip(user_end_positions[:-1], assistant_end_positions):
                        assistant_answer_mask[i][user_end+3:assistant_end-1] = 1
                    assistant_answer_mask[i][user_end_positions[-1]+3:] = 1
        
        return assistant_answer_mask

    def compute_loss(self, batch_data):
        """Compute loss for a batch of data"""
        # Forward pass
        logits = self.model(batch_data['x']).logits
        
        # Clear cache to manage memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Compute loss with masking
        log_probs = torch.log_softmax(logits, dim=-1)
        token_loss = torch.gather(
            log_probs, 
            dim=-1, 
            index=batch_data['y'].unsqueeze(2)
        ) * (-1)
        
        masked_loss = torch.mul(
            token_loss.squeeze(2), 
            batch_data['combined_mask']
        )
        
        # Compute average loss per sequence
        sequence_loss = masked_loss.sum(dim=-1) / batch_data['combined_mask'].sum(dim=-1)
        
        # Final loss with gradient accumulation
        loss = torch.nanmean(sequence_loss) / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss

    def compute_learning_rate(self, step, total_steps):
        """Compute learning rate with warmup and cosine decay"""
        if step < self.config.warmup_steps:
            # Linear warmup
            lr = self.config.max_lr * step / self.config.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.config.warmup_steps) / (total_steps - self.config.warmup_steps)
            decay = 0.5 * (1 + np.cos(np.pi * progress))
            lr = (self.config.max_lr - self.config.min_lr) * decay + self.config.min_lr
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

    def log_progress(self, step, train_loss_list):
        """Log training progress and metrics"""
        try:
            # Calculate average loss over the logging interval
            recent_losses = train_loss_list[-self.config.log_iter:]
            avg_loss = np.nanmean(recent_losses)
            
            # Create log message
            log_msg = (
                f"Time: {time.strftime('%Y-%m-%d, %H:%M:%S')}, "
                f"Step: {step + 1}, "
                f"Last_{self.config.log_iter}_steps_avg_loss: {avg_loss:.4f}"
            )
            
            # Log to console
            logger.info(log_msg)
            
            # Log to file
            self.log_to_file(step, avg_loss)
            
            # Optional: Add more metrics logging here if needed
            # For example: learning rate, gradient norms, etc.
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.2e}")
            
        except Exception as e:
            logger.error(f"Error in logging progress: {str(e)}")
            raise

    def log_to_file(self, step, avg_loss):
        """Log training metrics to a file"""
        log_file = os.path.join(self.config.output_dir, f"{self.config.model_name}-SFT_log.txt")
        
        # Create the log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Append log entry to file
        with open(log_file, "a") as f:
            log_entry = (
                f"Time: {time.strftime('%Y-%m-%d, %H:%M:%S')}, "
                f"Step: {step + 1}, "
                f"Avg_Loss: {avg_loss:.4f}\n"
            )
            f.write(log_entry)

    def initialize_logging(self):
        """Initialize logging configuration"""
        # Add initial training configuration to log file
        log_file = os.path.join(self.config.output_dir, f"{self.config.model_name}-SFT_log.txt")
        
        with open(log_file, "a") as f:
            f.write(
                f"=== Training Configuration ===\n"
                f"Time: {time.strftime('%Y-%m-%d, %H:%M:%S')}\n"
                f"Model: {self.config.model_name}\n"
                f"Batch size: {self.config.batch_size}\n"
                f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}\n"
                f"Max learning rate: {self.config.max_lr}\n"
                f"Min learning rate: {self.config.min_lr}\n"
                f"Warmup steps: {self.config.warmup_steps}\n"
                f"Trainable parameters: {self.trainable_params:,}\n"
                f"============================\n\n"
            )
    def train(self):
        """Main training loop"""
        dataset = self.prepare_dataset()
        total_steps = len(dataset['train']) // self.config.batch_size
        
        self.model.train()
        train_loss_list = []
        ignore_iters_count = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step in range(total_steps):
                try:
                    loss = self.training_step(dataset, step)
                    train_loss_list.append(loss)
                    
                    if self.should_log(step):
                        self.log_progress(step, train_loss_list)
                    
                    if self.should_save(step):
                        self.save_model(epoch, step)
                        
                except Exception as e:
                    logger.error(f"Error in training step {step}: {e}")
                    ignore_iters_count += 1
                    continue

        logger.info(f"Training completed. Ignored {ignore_iters_count} batches.")
        self.save_model(epoch, step, is_final=True)

    def training_step(self, dataset, step):
        """Execute single training step"""
        try:
            batch_data = self.prepare_batch(dataset, step)
            
            # Update learning rate
            total_steps = len(dataset['train']) // self.config.batch_size
            self.compute_learning_rate(step, total_steps)
            
            # Compute loss
            loss = self.compute_loss(batch_data)
            
            # Optimizer step if needed
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in training step {step}: {str(e)}")
            raise

    def should_log(self, step):
        """Determine if current step should log progress"""
        return (step + 1) % self.config.log_iter == 0

    def should_save(self, step):
        """Determine if current step should save model"""
        return ((step + 1) * self.config.batch_size) % self.config.save_steps == 0

    def save_model(self, epoch, step, is_final=False):
        """Save model checkpoint"""
        suffix = "final" if is_final else f"epoch{epoch}_step{step}"
        save_path = os.path.join(
            self.config.output_dir,
            f"{self.config.model_name}_SFT_{suffix}.pth"
        )
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

def main():
    config = TrainingConfig()
    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()