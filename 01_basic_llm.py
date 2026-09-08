from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain quantum computing to a 10-year-old."
)

print(response.text)