import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

name = "google/gemma-2-9b-it"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")

prompt = """
Please put the following dates in order:

August 2nd, March 15, December 21, October 31, June 21, July 20"
"""

messages = [{"role": "user", "content": prompt}]
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
    eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")],
)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
