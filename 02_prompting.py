from google import genai
from dotenv import load_dotenv

load_dotenv(override = True)

client = genai.Client()

prompt = """
Classify the following sentence as positive or negative.

Sentence:
"I really enjoyed this movie."

"""
response = client.models.generate_content(
    model = "gemini-3.6-flash",
    contents = prompt
)

print(response.text)