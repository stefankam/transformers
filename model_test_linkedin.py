import torch
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

# Load a configuration
model = T5ForConditionalGeneration.from_pretrained("t5-small")
# Use a base model to load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Save it so that from_pretrained() works
model.save_pretrained("/home/skb67/transformers/src/transformers/models/t5_dynamic")
# Save tokenizer to the custom directory
tokenizer.save_pretrained('/home/skb67/transformers/src/transformers/models/t5_dynamic')


# Load T5 model and tokenizer
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

# Function to update encoder state dynamically
def update_encoder(comment):
    global encoder_outputs
    input_text = linkedin_post + "\n" + comment  # Append new comment
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    # Update encoder output without resetting previous state
    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

# Function to generate response dynamically
def generate_response():
    global decoder_input_ids
    with torch.no_grad():
        output = model.generate(
            decoder_input_ids=decoder_input_ids,  # Continue from previous response
            encoder_outputs=encoder_outputs,  # Use updated encoder context
            max_length=150
        )
    # Append new tokens to decoder input for continuous generation
    new_response = tokenizer.decode(output[0], skip_special_tokens=True)
    decoder_input_ids = tokenizer(new_response, return_tensors="pt").input_ids
    return new_response

# Simulate evolving comments in real-time
for new_comment in review_comments:
    update_encoder(new_comment)  # Update encoder with new context
    response = generate_response()  # Decoder updates dynamically
    print(f"\nNew comment: {new_comment}")
    print("Updated Response:", response)
