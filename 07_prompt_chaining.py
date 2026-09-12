from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

client = genai.Client()

review = """
The laptop looks great and performance is excellent,
but the battery lasts only 3 hours and the keyboard
feels uncomfortable.
"""

# Step 1: Extract key points
response1 = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=f"""
Extract the positive and negative aspects from this product review.

Review:
{review}

Return only the extracted information.
"""
)

extracted_info = response1.text

print("STEP 1 - EXTRACTED INFORMATION")
print(extracted_info)

# Step 2: Analyze sentiment
response2 = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=f"""
Analyze the following extracted product-review information.

{extracted_info}

Determine:
1. Overall sentiment
2. Whether the product appears recommendable
3. A rating from 1 to 5
"""
)

analysis = response2.text

print("\nSTEP 2 - ANALYSIS")
print(analysis)

# Step 3: Generate a customer-friendly summary
response3 = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=f"""
Write a short customer-friendly summary based on this analysis:

{analysis}

Keep it under 3 sentences.
"""
)

print("\nSTEP 3 - FINAL SUMMARY")
print(response3.text)