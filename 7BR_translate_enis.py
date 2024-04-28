import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Load base model and LoRA weights
model = AutoModelForCausalLM.from_pretrained("haoranxu/ALMA-7B-R", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("haoranxu/ALMA-7B-R", padding_side='left')

def translate_enis(en_sent):
    # Add the source sentence into the prompt template
    prompt = "Translate this from English to Icelandic:\nEnglish:" + en_sent + "\nIcelandic:"
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=100, truncation=True).input_ids.cuda()
    # Translation
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=80, do_sample=True,
                                       temperature=0.6, top_p=0.9, top_k=0)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    try:
        return outputs[0].split('\nIcelandic:')[1].lstrip().strip()
    except:
        return outputs

while True:
    try:
        user_input = input("Write a sentence for me to translate from English to Icelandic: ")
        print(translate_enis(user_input))
    except KeyboardInterrupt:
        break
