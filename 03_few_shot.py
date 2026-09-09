from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

client = genai.Client()

prompt = """
Classify the qery into the following categories into technical,billing,account,general

Examples:
"i have a order issue, when can i call customer support?" - general.
"i am a active user but my accounts seems to be hached." - technical
"the bill showed 1k but 1.5k was debited from my account" - billing
"forgot the location of bank where i have account, pls give my account details" - account

"I cannot log into my account."
"I was charged twice for my subscription." 
"The application crashes whenever I upload a file." 
"What are your customer support hours?"
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print(response.text)