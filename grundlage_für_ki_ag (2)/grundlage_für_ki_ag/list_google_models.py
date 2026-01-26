import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Verfügbare Modelle:")
for m in genai.list_models():
    print(f"- {m.name}")