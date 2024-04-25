from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the pad_token_id in the model configuration
model.config.pad_token_id = model.config.eos_token_id

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get the character description and prompt from the user
character_desc = input("Enter the character description: ")
prompt = input("Enter the initial prompt: ")

# Combine the character description and the prompt
input_text = character_desc + "\n\n" + prompt

while True:
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate the attention mask
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float()

    # Generate text
    output = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7, attention_mask=attention_mask)

    # Decode the output
    output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    print(output_text)

    # Get the next input from the user
    next_input = input("Enter the next input (or 'quit' to exit): ")

    # Check if the user wants to quit
    if next_input.lower() == 'quit':
        break

    # Update the input text
    input_text = output_text + "\n\n" + next_input