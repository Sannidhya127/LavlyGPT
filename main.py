from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from colored import fg, attr
import re

# Initialize the GPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.config.pad_token_id = tokenizer.eos_token_id
model.config.padding_side = 'left'
# Initialize the intent classification model
classifier = pipeline("text-classification", model="textattack/bert-base-uncased-imdb")

# Initialize chat history
chat_history_ids = None

while True:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    # Get user input
    user_input = input(f"{fg('green_1')}>> User: {attr('reset')}")

    # Check if the user wants to quit
    if user_input.lower() == "quit":
        break

    # Check if the user input is a mathematical expression
    if re.match("^[0-9+\-*/() ]+$", user_input):
        print("LavlyGPT: " + str(eval(user_input)))
        continue

    # Classify the intent of the user's question
    intent = classifier(user_input)[0]

    # Check if the user is asking for the bot's name
    if intent['label'] == 'POSITIVE' and 'name' in user_input.lower():
        print("LavlyGPT: My name is Lavly Bhai")
        continue

    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    model.config.padding_side = 'left'
    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6)

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    model.config.padding_side = 'left'
    # Print the response
    print(f"{fg('red_1')}LavlyGPT: {response}{attr('reset')}")