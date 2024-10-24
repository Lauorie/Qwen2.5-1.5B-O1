from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/root/app/Reason/checkpoints"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from typing import List, Dict
def new_apply_chat_template(history:List[Dict[str, str]], add_reasoning_generation_prompt:bool=True, add_assistant_generation_prompt:bool=False):
  if add_reasoning_generation_prompt:
    return "".join([f"<|im_start|>{i['role']}\n{i['content']}<|im_end|>\n" for i in history]) + "<|im_start|><|reasoning|>\n"
  if add_assistant_generation_prompt:
    return "".join([f"<|im_start|>{i['role']}\n{i['content']}<|im_end|>\n" for i in history]) + "<|im_start|>assistant\n"
  

from IPython.display import Markdown, display
device = "cuda"
history = []
history.append({"role": "system", "content": "You are a helpful assistant"})
while True:
    question = input('User：' + '\n')
    print(question)
    print('\n')
    history.append({"role": "user", "content": question})

    input_text = new_apply_chat_template(
            history,
            add_reasoning_generation_prompt=True
        )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

    if model_inputs.input_ids.size()[1]>32000:
        break

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=3000
    )

    if len(generated_ids)>32000:
        break

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    reasoning_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history.append({"role": "<|reasoning|>", "content": reasoning_response})
    print('reasoning:\n')
    #print(response)
    display(Markdown(reasoning_response))
    print("------------")
    print('\n')

    input_text = new_apply_chat_template(
            history,
            add_assistant_generation_prompt=True
        )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

    if model_inputs.input_ids.size()[1]>32000:
        break

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=3000
    )

    if len(generated_ids)>32000:
        break

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    assistant_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history.append({"role": "assistant", "content": assistant_response})
    print('assistant:\n')
    display(Markdown(assistant_response))
    print("------------")

print("超过模型字数上线，已退出")
