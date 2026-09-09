from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain SQL joins.",
    config=types.GenerateContentConfig(
        system_instruction="You are a senior database architect."
    )
)

print(response.text)