from google import genai
from dotenv import load_dotenv

load_dotenv(override = True)

client = genai.Client()

prompt = """
Classify the following sentence as positive or negative.

Sentence:
"The product broke after two days and I am very disappointed."

"""
response = client.models.generate_content(
    model = "gemini-3.6-flash",
    contents = prompt
)

print(response.text)