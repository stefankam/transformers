import threading
import time
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Sample LinkedIn Post
linkedin_post = "This is a post about the importance of cybersecurity in today's digital world. How do you protect your data online?"

# Simulated evolving review comments
review_comments = [
    "I think the best way is using a combination of firewalls and encryption.",
    "Absolutely! Also, it's important to use strong passwords and multi-factor authentication.",
    "Great post! The role of AI in cybersecurity is really growing, and it's exciting to see how technology is evolving.",
    "Another important aspect is user education. People need to be aware of phishing attacks.",
    "Cloud security is also crucial. Many businesses are moving to the cloud, so security strategies should adapt accordingly."
]

# Load model and tokenizer
model_name = "/home/skb67/transformers/src/transformers/models/t5_dynamic"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Tokenize the initial LinkedIn post
inputs = tokenizer(linkedin_post, return_tensors="pt", padding=True, truncation=True)

# Encode the post
with torch.no_grad():
    encoder_outputs = model.encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

# Initialize decoder with "Response:"
decoder_input_ids = tokenizer("Response:", return_tensors="pt").input_ids

# Track last encoder output for comparison
last_encoder_output = encoder_outputs.last_hidden_state.clone()

# Locks to protect encoder-decoder data
encoder_lock = threading.Lock()
decoder_lock = threading.Lock()

# Flag to indicate new encoder update
encoder_updated = False

# Function to update encoder state with comments
def update_encoder(comment):
    global encoder_outputs, last_encoder_output, encoder_updated
    
    input_text = linkedin_post + "\n" + comment
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    with encoder_lock:
        with torch.no_grad():
            encoder_outputs = model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # Compare encoder outputs
            new_state = encoder_outputs.last_hidden_state
            old_state = last_encoder_output
            min_len = min(new_state.shape[1], old_state.shape[1])
            new_state = new_state[:, :min_len, :]
            old_state = old_state[:, :min_len, :]
            diff = (new_state - old_state).abs().mean()
            print(f"Encoder output change (L1 norm difference): {diff.item():.6f}")

            if diff.item() > 0.01:  # Only mark as updated if significant
                encoder_updated = True
            last_encoder_output = encoder_outputs.last_hidden_state.clone()

# Function to simulate new comments arrival
def simulate_comments():
    for comment in review_comments:
        time.sleep(5)  # Simulate new comment every 5 seconds
        print("\n" + "=" * 50)
        print(f"New comment received: {comment}")
        update_encoder(comment)

# Function to generate response
def generate_response():
    global decoder_input_ids, encoder_updated
    while True:
        time.sleep(1)  # Simulate response every second
        with encoder_lock:
            if encoder_updated:
                print("[INFO] Encoder updated. Decoder will attend to the new context.")
                encoder_updated = False  # Reset flag

        with torch.no_grad():
            output = model.generate(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                max_length=150
            )
        
        # Decode response
        new_response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated response: {new_response}")
        
        response_tokens = tokenizer.encode(new_response, return_tensors="pt")
        decoder_input_ids = response_tokens

# Start concurrent threads
comment_thread = threading.Thread(target=simulate_comments)
response_thread = threading.Thread(target=generate_response)

comment_thread.start()
response_thread.start()

# Wait for both threads to complete
comment_thread.join()
response_thread.join()
