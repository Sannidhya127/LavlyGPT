from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from colored import fg, attr
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.config.pad_token_id = tokenizer.eos_token_id
model.config.padding_side = 'left'

# Initialize chat history
chat_history_ids = None

while True:
    # Get user input
    user_input = input(f"{fg('red_1')}>> User:{attr('reset')}")
        
    # Check if the user wants to quit
    if user_input.lower() == "quit":
        break

    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.8)

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Print the response
    print(f"{fg('green_1')}LavlyGPT: {response}{attr('reset')}")