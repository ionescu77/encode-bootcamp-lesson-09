import torch
import datetime

from transformers import pipeline

# device = torch.device("mps")      # not needed!?

# Record start time
start_time = datetime.datetime.now()
print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

#pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

input_text = [
 {
     "role": "system",
     "content": "You are a friendly chatbot who always responds like an Italian chef",
 },
 {"role": "user", "content": "What is the best recipe for Pepperoni pizza?"},
]

prompt = pipe.tokenizer.apply_chat_template(input_text, tokenize=False, add_generation_prompt=True)

outputs = pipe(prompt, max_new_tokens=256)

print(outputs[0]["generated_text"])

# Record end time
end_time = datetime.datetime.now()
print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))

# Calculate duration
duration = end_time - start_time
duration_seconds = duration.total_seconds()
duration_h = int(duration_seconds // 3600)
duration_m = int((duration_seconds % 3600) // 60)
duration_s = int(duration_seconds % 60)

print(f"Duration: {duration_h}h:{duration_m}m:{duration_s}s")
