# Dynamic Attention Changes for T5 Model

## Introduction

In this project, we explore changes made to the `T5ForConditionalGeneration` model and `T5Attention` in Hugging Face's Transformers library. The aim is to implement dynamic attention updates based on encoder changes and evolving context during autoregressive generation. The following sections summarize the changes made, how they affect the model's behavior, and how the impact of these changes can be verified through dynamic updates during inference.

## Key Changes to the T5 Model

### 1. Changes to `T5ForConditionalGeneration.forward`

The primary change in the `forward` function of `T5ForConditionalGeneration` was to update the encoder outputs dynamically as new context (in the form of comments) is provided.


**Original Code:**
```
if encoder_outputs is None:
    encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
encoder_hidden_states = encoder_outputs[0]
```

is modified to:

```
if encoder_outputs is None:
    new_encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    new_encoder_hidden_states = new_encoder_outputs[0]

    # Compute difference between old and new encoder outputs
    if self.past_encoder_hidden_states is not None:
        diff = torch.norm(new_encoder_hidden_states - self.past_encoder_hidden_states, dim=-1).mean()
        if diff > self.update_threshold:  # Threshold to determine significant change
            self.past_encoder_hidden_states = new_encoder_hidden_states
            encoder_hidden_states = new_encoder_hidden_states
        else:
            encoder_hidden_states = self.past_encoder_hidden_states  # Keep previous states
    else:
        self.past_encoder_hidden_states = new_encoder_hidden_states
        encoder_hidden_states = new_encoder_hidden_states
```

Compute a similarity score between the current and previous encoder hidden states.

If the difference exceeds a threshold, update the KV pairs in the decoder.


### 2. Modifications to T5Attention

The changes to T5Attention were focused on updating the attention mechanism during autoregressive generation. Specifically, the decoder attends to the encoder's output at each step, considering both previous and new key-value states.

Modified Code in T5Attention.forward:

```
current_states = key_value_states if is_cross_attention else hidden_states
if is_cross_attention and past_key_value is not None:
    if is_updated and not self.encoder_has_changed:  # Only reuse cache if encoder is unchanged
        key_states = curr_past_key_value.key_cache[self.layer_idx]
        value_states = curr_past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self.k(key_value_states)
        value_states = self.v(key_value_states)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # Update KV cache since encoder changed
        past_key_value.is_updated[self.layer_idx] = True

    if past_key_value is not None:
        # save all key/value_states to cache to be re-used for fast auto-regressive generation
        cache_position = cache_position if not is_cross_attention else None
        key_states, value_states = curr_past_key_value.update(
            key_states, value_states, self.layer_idx, {"cache_position": cache_position}
        )
        # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
        if is_cross_attention:
            past_key_value.is_updated[self.layer_idx] = True
``` 

### 3. Dynamic Response Generation

The response generation is dynamically updated with new comments in the context of previous user input. This ensures that the generated response evolves based on the accumulated context:

generate_response Function: Each time a new comment is passed, the encoder is updated with the new comment, and the decoder generates a response based on the updated context. The response is then used as part of the input for the next generation step, ensuring autoregressive behavior.


``` 
def generate_response():
    global decoder_input_ids
    output = model.generate(
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        max_length=150
    )
    new_response = tokenizer.decode(output[0], skip_special_tokens=True)
    decoder_input_ids = tokenizer(new_response, return_tensors="pt").input_ids
    return new_response

``` 

### 4. Changes to Model Inputs and Outputs

Encoder Input Handling: The model now takes dynamic input sequences, where decoder_input_ids are updated based on evolving context, and the model recomputes the encoder outputs after each new comment.

Decoder Handling: The decoder uses the updated encoder outputs and previously generated tokens as input to generate responses.

``` 
# Simulate evolving comments in real-time
for new_comment in review_comments:
    update_encoder(new_comment)  # Update encoder with new context
    response = generate_response()  # Decoder updates dynamically
    print(f"\nNew comment: {new_comment}")
    print("Updated Response:", response)
``` 



## How to Use
Install Dependencies: Ensure that you have the required libraries installed, such as torch and transformers by Hugging Face.
``` 
pip install torch transformers
``` 

If you choose to test the model t5\_dynamic with linkedin data, you could  use the file model\_test\_linkedin\_concurrent.py, where

Model Initialization: Initialize the model and tokenizer from the pre-trained T5 model:

``` 
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
``` 

Input Preparation: Define the initial post and simulated evolving review comments:

``` 
linkedin_post = "This is a post about the importance of cybersecurity in today's digital world."
review_comments = [
    "I think the best way is using a combination of firewalls and encryption.",
    "Absolutely! Also, it's important to use strong passwords and multi-factor authentication.",
    "Great post! The role of AI in cybersecurity is really growing.",
    "Another important aspect is user education. People need to be aware of phishing attacks.",
    "Cloud security is also crucial."
]
``` 

Run the Simulation: The code will simulate evolving comments, updating the encoder's context, and generating responses dynamically.

``` 
for new_comment in review_comments:
    update_encoder(new_comment)
    response = generate_response()
    print(f"New comment: {new_comment}")
    print(f"Generated response: {response}")
``` 

There are the following changes in terms of concurrency with respect to model\_test\_linkedin.py,

```
Concurrency with Threads:

- comment_thread: Simulates the arrival of comments by calling update_encoder.

- response_thread: Continuously generates responses using generate_response.

Encoder Update with Locks:

- encoder_lock: Prevents race conditions when reading or updating encoder_outputs.

- encoder_updated: A flag that signals when the decoder should fetch new encoder outputs.

Efficient Update Check:

- Compares the last encoder output with the new one using L1 norm.

- Updates only if the difference is greater than a small threshold (e.g., 0.01).

Real-time Response Generation:

- Decoder attends to updated encoder outputs in real-time when flagged as updated.
```

