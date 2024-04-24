import openai

openai.api_key = 'sk-proj-ENUJa5XWN5Xf5mwniw0AT3BlbkFJxN9q5Pe3WN1Z0zJRflAd'

response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="Translate the following English text to French: '{}'",
  max_tokens=60
)

print(response.choices[0].text.strip())