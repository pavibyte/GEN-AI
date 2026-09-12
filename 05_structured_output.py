from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

class ReviewAnalysis(BaseModel):
    sentiment: str
    rating: int
    positive_aspects: list[str]
    negative_aspects: list[str]


client = genai.Client()

prompt = """
Analyze this product review:

"The phone has an excellent camera but poor battery life."
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ReviewAnalysis,
    ),
)

print(response.text)