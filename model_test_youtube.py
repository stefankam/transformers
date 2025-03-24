import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load conversation from file
with open('./data/conversation.txt', 'r') as file:
    conversation = file.read().split('\n')

# Load modified model and tokenizer
model_name = "/home/skb67/transformers/src/transformers/models/t5_dynamic"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Tokenize the initial conversation
initial_input = tokenizer(conversation[0], return_tensors="pt", padding=True, truncation=True)

# Encode the initial input
with torch.no_grad():
    encoder_outputs = model.encoder(
        input_ids=initial_input["input_ids"],
        attention_mask=initial_input["attention_mask"]
    )

# Initialize decoder with "Response:"
decoder_input_ids = tokenizer("Response:", return_tensors="pt").input_ids

# Track last encoder output for comparison
last_encoder_output = encoder_outputs.last_hidden_state.clone()

# Function to update encoder state dynamically and log changes
def update_encoder(comment):
    global encoder_outputs, last_encoder_output
    input_text = '\n'.join(conversation[:conversation.index(comment)+1])
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    # Compute L1 norm difference
    new_state = encoder_outputs.last_hidden_state
    old_state = last_encoder_output
    min_len = min(new_state.shape[1], old_state.shape[1])
    diff = (new_state[:, :min_len, :] - old_state[:, :min_len, :]).abs().mean()
    print(f"Encoder output change (L1 norm difference): {diff.item():.6f}")
    last_encoder_output = encoder_outputs.last_hidden_state.clone()

# Function to generate response dynamically and save to file
def generate_response():
    global decoder_input_ids
    with torch.no_grad():
        output = model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=150
        )
    new_response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated response: {new_response}")

    # Save response to file
    with open('./data/response.txt', 'a') as f:
        f.write(new_response + '\n')

    decoder_input_ids = tokenizer.encode(new_response, return_tensors="pt")

# Simulate the conversation processing
for comment in conversation[1:]:
    print("\n" + "=" * 50)
    print(f"New comment: {comment}")
    update_encoder(comment)
    generate_response()
