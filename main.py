from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
from transformers import Trainer, TrainingArguments

from torch.utils.data import Dataset
# Training dialogues
train_dialogues = [
    ("What is the formula for the area of a circle?", "The formula for the area of a circle is πr^2, where r is the radius."),
    ("Can you explain Newton's laws of motion?", "Sure! Newton's laws of motion describe the relationship between the motion of an object and the forces acting on it."),
    ("How do I solve a quadratic equation?", "To solve a quadratic equation, you can use the quadratic formula or factorization method."),
    ("What is the difference between velocity and speed?", "Velocity is a vector quantity that includes both magnitude and direction, while speed is a scalar quantity that only includes magnitude."),
    ("Can you help me understand Ohm's law?", "Ohm's law states that the current flowing through a conductor is directly proportional to the voltage across it, given a constant resistance."),
    ("How can I balance chemical equations?", "To balance chemical equations, you need to ensure that the number of atoms of each element is the same on both sides of the equation."),
    ("What is the concept of differentiation in calculus?", "Differentiation is a mathematical process used to find the rate at which a function changes."),
    ("Can you explain the concept of inertia?", "Inertia is the resistance of an object to change its state of motion."),
    ("How does the periodic table work?", "The periodic table organizes elements based on their atomic number, electron configuration, and recurring chemical properties."),
    ("What is the difference between covalent and ionic bonds?", "Covalent bonds involve the sharing of electrons between atoms, while ionic bonds involve the transfer of electrons from one atom to another.")
]

# Test dialogues
test_dialogues = [
    ("What are the three laws of thermodynamics?", "The three laws of thermodynamics describe the behavior of thermodynamic systems and energy transfer."),
    ("How do I calculate the pH of a solution?", "The pH of a solution can be calculated using the formula pH = -log[H+], where [H+] is the concentration of hydrogen ions in the solution."),
    ("Can you explain the concept of vector addition?", "Vector addition is the process of combining two or more vectors to find their resultant vector."),
    ("What are the different types of chemical reactions?", "The different types of chemical reactions include synthesis, decomposition, single replacement, double replacement, and combustion reactions."),
    ("How does the greenhouse effect work?", "The greenhouse effect is the process by which greenhouse gases trap heat from the sun in the Earth's atmosphere, leading to a warming effect."),
    ("Can you help me understand the concept of wave interference?", "Wave interference occurs when two or more waves overlap and combine to form a new wave."),
    ("What is the concept of equilibrium in chemistry?", "Equilibrium in chemistry refers to a state in which the forward and reverse reactions of a chemical reaction occur at the same rate."),
    ("How can I calculate the force of gravity between two objects?", "The force of gravity between two objects can be calculated using Newton's law of universal gravitation."),
    ("Can you explain the concept of half-life in radioactive decay?", "The half-life of a radioactive substance is the time it takes for half of the atoms in a sample to decay."),
    ("What is the difference between an exothermic and endothermic reaction?", "An exothermic reaction releases heat to its surroundings, while an endothermic reaction absorbs heat from its surroundings.")
]

# Additional training dialogues
train_dialogues.extend([
    ("How's the weather today?", "The weather is sunny with a few clouds."),
    ("What did you do over the weekend?", "I spent the weekend studying for exams and hanging out with friends."),
    ("Do you have any plans for the holidays?", "I'm planning to travel and visit some family members."),
    ("What's your favorite movie?", "I enjoy watching science fiction movies like Interstellar."),
    ("Can you recommend a good book to read?", "Sure! I recommend 'The Martian' by Andy Weir."),
    ("What's the capital of France?", "The capital of France is Paris."),
    ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming."),
    ("What's the largest organ in the human body?", "The largest organ in the human body is the skin."),
    ("How do plants make food?", "Plants make food through a process called photosynthesis, where they use sunlight to convert carbon dioxide and water into glucose and oxygen."),
    ("What causes earthquakes?", "Earthquakes are caused by the sudden release of energy in the Earth's crust."),
    ("Who developed the theory of relativity?", "The theory of relativity was developed by Albert Einstein."),
    ("Can you explain the concept of supply and demand?", "Supply and demand is an economic model that describes the relationship between the availability of a product and its demand by consumers."),
    ("How do I calculate the volume of a sphere?", "The volume of a sphere can be calculated using the formula V = (4/3)πr^3, where r is the radius of the sphere."),
    ("What is the structure of an atom?", "An atom consists of a nucleus containing protons and neutrons, surrounded by electrons orbiting in energy levels."),
    ("What is the process of mitosis?", "Mitosis is the process of cell division in which a single cell divides into two identical daughter cells."),
    ("Can you explain the concept of continental drift?", "Continental drift is the theory that the Earth's continents have moved relative to each other over geological time scales."),
    ("How do I solve a system of linear equations?", "A system of linear equations can be solved using methods like substitution, elimination, or matrices."),
    ("What is the formula for the area of a triangle?", "The formula for the area of a triangle is (1/2) * base * height."),
    ("Can you explain the concept of osmosis?", "Osmosis is the movement of water molecules through a selectively permeable membrane from an area of higher concentration to an area of lower concentration."),
    ("Who is credited with discovering the structure of DNA?", "The structure of DNA was discovered by James Watson and Francis Crick."),
])

# Additional test dialogues
test_dialogues.extend([
    ("What are the symptoms of COVID-19?", "Common symptoms of COVID-19 include fever, cough, and difficulty breathing."),
    ("How do I calculate the area of a trapezoid?", "The area of a trapezoid can be calculated using the formula A = ((a + b) / 2) * h, where 'a' and 'b' are the lengths of the parallel sides and 'h' is the height."),
    ("Can you explain the process of meiosis?", "Meiosis is a type of cell division that results in four daughter cells with half the number of chromosomes as the parent cell."),
    ("What is the difference between an acid and a base?", "An acid is a substance that donates protons, while a base is a substance that accepts protons."),
    ("How does the human respiratory system work?", "The human respiratory system is responsible for exchanging oxygen and carbon dioxide between the body and the environment."),
    ("What is the process of photosynthesis?", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."),
    ("Can you explain the concept of buoyancy?", "Buoyancy is the ability of an object to float in a fluid, which depends on the object's density and the density of the fluid."),
    ("Who proposed the theory of evolution?", "The theory of evolution was proposed by Charles Darwin."),
    ("What is the difference between renewable and non-renewable resources?", "Renewable resources are resources that can be replenished naturally over time, while non-renewable resources are finite and cannot be replenished once depleted."),
    ("How do I calculate the area of a rectangle?", "The area of a rectangle can be calculated by multiplying its length by its width."),
    ("Can you explain the concept of inertia?", "Inertia is the resistance of an object to changes in its state of motion."),
    ("What is the difference between velocity and acceleration?", "Velocity is the rate of change of displacement with respect to time, while acceleration is the rate of change of velocity with respect to time."),
    ("How do I calculate the kinetic energy of an object?", "The kinetic energy of an object can be calculated using the formula KE = (1/2) * m * v^2, where 'm' is the mass of the object and 'v' is its velocity."),
    ("What is the process of cellular respiration?", "Cellular respiration is the process by which cells break down glucose and other organic molecules to produce ATP, the primary energy carrier in cells."),
    ("Can you explain the concept of magnetism?", "Magnetism is a force that attracts or repels certain materials, such as iron or steel."),
    ("Who is considered the father of modern physics?", "Albert Einstein is often considered the father of modern physics."),
    ("What is the difference between a conductor and an insulator?", "A conductor is a material that allows the flow of electrical charge, while an insulator is a material that does not allow the flow of electrical charge."),
    ("How do I calculate the perimeter of a rectangle?", "The perimeter of a rectangle can be calculated by adding together the lengths of all four sides."),
    ("Can you explain the concept of centripetal force?", "Centripetal force is a force that acts on an object moving in a circular path and is directed towards the center of the circle."),
    ("What is the process of nuclear fusion?", "Nuclear fusion is the process by which multiple atomic nuclei collide and fuse together to form a heavier nucleus, releasing energy in the process."),
])
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
# from transformers import Trainer, TrainingArguments

# # Initialize the GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token

# # Prepare the training and test dialogues
# train_dialogues = [dialogue for dialogue, _ in train_dialogues]
# test_dialogues = [dialogue for dialogue, _ in test_dialogues]

# # Encode the dialogues
# train_encodings = tokenizer(train_dialogues, truncation=True, padding=True)
# test_encodings = tokenizer(test_dialogues, truncation=True, padding=True)
# from datasets import Dataset

# # Prepare the datasets
# train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids']})
# test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids']})
# # Prepare the datasets
# # train_dataset = TextDataset(train_encodings)
# # test_dataset = TextDataset(test_encodings)

# # Define the data collator
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Define the training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
# )

# # Initialize the trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# Train the model
# trainer.train()

# class ChatDataset(Dataset):
#     def __init__(self, tokenizer, dialogues, max_length=512):
#         self.tokenizer = tokenizer
#         self.dialogues = dialogues
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.dialogues)

#     def __getitem__(self, i):
#         dialogue = self.dialogues[i]
#         return self.tokenizer(dialogue, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
# )


# tokenizer.pad_token = tokenizer.eos_token

# # Create trainer
# trainer = Trainer(
#     model=model,                        
#     args=training_args,                  
#     train_dataset = ChatDataset(tokenizer, train_dialogues),
#     eval_dataset = ChatDataset(tokenizer, test_dialogues)         
# )

# # Train the model
# trainer.train()
# trainer.save_model('./results')
while True:
    # Get user input
    user_input = input(">> User:")
        
    # Check if the user wants to quit
    if user_input.lower() == "quit":
        break

    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if 'chat_history_ids' in locals() else new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Pretty print last output tokens from bot
    print("LavlyGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the trained model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained('./results')
# model = AutoModelForCausalLM.from_pretrained('./results')

# # Function to let the model generate a response
# def get_response(input_text):
#     # Encode the input text
#     input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

#     # Generate a response
#     response_ids = model.generate(input_ids, max_length=512, do_sample=True)

#     # Decode the response
#     response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

#     return response

# # Test the function
# input_text = input("USER>>")
# print(get_response(input_text))