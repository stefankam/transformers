
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

# Load modified model and tokenizer
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


# Function to update encoder state dynamically and log changes
def update_encoder(comment):
    global encoder_outputs, last_encoder_output
    input_text = linkedin_post + "\n" + comment  # Append new comment
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    # Align tensor shapes (pad or truncate)
    new_state = encoder_outputs.last_hidden_state
    old_state = last_encoder_output
    min_len = min(new_state.shape[1], old_state.shape[1])
    new_state = new_state[:, :min_len, :]
    old_state = old_state[:, :min_len, :]
    # Compute L1 norm difference
    diff = (new_state - old_state).abs().mean()
    print(f"Encoder output change (L1 norm difference): {diff.item():.6f}")
    # Store current encoder output for next comparison
    last_encoder_output = encoder_outputs[0].clone()


# Function to extract and log attention weights
def get_attention_weights():
    with torch.no_grad():
        # Forward pass with hooks to capture attention
        def hook_fn(module, input, output):
            global attention_weights
            attention_weights = output
        attention_layer = model.encoder.block[0].layer[0].SelfAttention
        hook = attention_layer.register_forward_hook(hook_fn)
        # Re-run encoder to capture attention weights
        model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        # Remove hook after capturing
        hook.remove()
        # Compute mean attention weights over heads
        avg_attention = attention_weights[0].mean(dim=1)  # Average over attention heads
        print(f"Avg attention weights (first token): {avg_attention[0, 0].detach().cpu().numpy()}")


# Function to generate response dynamically and compare results
def generate_response():
    global decoder_input_ids
    with torch.no_grad():
        output = model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=150
        )
    # Append new tokens to decoder input for continuous generation
    new_response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated response: {new_response}")
    # Compare new decoder output with previous one
    response_tokens = tokenizer.encode(new_response, return_tensors="pt")
    #if decoder_input_ids is not None:
    #    diff = F.mse_loss(response_tokens.float(), decoder_input_ids.float()).item()
    #    print(f"Decoder response change (MSE loss): {diff:.6f}")
    decoder_input_ids = response_tokens
    return new_response


# Simulate evolving comments in real-time
for new_comment in review_comments:
    print("\n" + "=" * 50)
    print(f"New comment: {new_comment}")
    update_encoder(new_comment)  # Update encoder with new context
    get_attention_weights()  # Log attention changes
    response = generate_response()  # Decoder updates dynamically

