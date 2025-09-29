import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # If the API key is not found, raise an error with instructions
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Please create a .env file in the same directory as this script "
        "and add the line: GOOGLE_API_KEY='your_api_key_here'"
    )

try:
    # Configure the genai library with the API key
    genai.configure(api_key=api_key)

    print("Successfully configured API key.")
    print("-" * 30)
    print("Available models supporting 'generateContent':")
    
    # List all models and filter for the ones that can generate content
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check if your API key is valid and has the necessary permissions.")