import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Load base model and LoRA weights
model = AutoModelForCausalLM.from_pretrained("haoranxu/ALMA-7B-R", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("haoranxu/ALMA-7B-R", padding_side='left')


def translate_isen(is_sent):
    # Add the source sentence into the prompt template
    prompt = "Translate this from Icelandic to English:\nIcelandic:" + is_sent + "\nEnglish:"
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=100, truncation=True).input_ids.cuda()
    # Translation
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=80, do_sample=True,
                                       temperature=0.6, top_p=0.9)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    try:
        return outputs[0].split('\nEnglish:')[1].strip().lstrip()
    except:
        return outputs

while True:
    try:
        user_input = input("Skrifaðu setningu fyrir mig að þýða úr íslenska á ensku: ")
        print(translate_isen(user_input))
    except KeyboardInterrupt:
        break
