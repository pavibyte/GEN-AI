from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override =True)

client = genai.Client()

prompt = """
Give me 5 creative names for an AI-powered fitness application.
Return only the names.
"""
#ax_output_tokens = 300
for temperature in [0.0, 0.5, 1.0]:
    print(f"\n --- Temperature: {temperature} ---")

    response =client.models.generate_content(
        model = "gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )
    print(response.text)