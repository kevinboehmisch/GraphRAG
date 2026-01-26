import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Umweltvariablen laden
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ FEHLER: GOOGLE_API_KEY wurde nicht in der .env gefunden!")
    exit()

print(f"🔑 Key gefunden (endet auf ...{api_key[-4:]})")

# 2. Google SDK konfigurieren
genai.configure(api_key=api_key)

# 3. Test-Anfrage
try:
    # Wir nehmen 1.5-flash, das ist der Standard für schnelle Tests
    model = genai.GenerativeModel('models/gemma-3-27b-it')
    
    print("🚀 Sende Test-Anfrage an Gemini...")
    response = model.generate_content("Antworte kurz mit dem Wort 'BEREIT'.")
    
    print(f"\n✅ ERFOLG! Gemini sagt: {response.text}")

except Exception as e:
    print("\n❌ API-FEHLER DETAILS:")
    error_msg = str(e)
    
    if "429" in error_msg:
        print("🛑 STATUS: Quota Exceeded (429). Dein Limit für diesen Zeitraum ist voll.")
    elif "400" in error_msg:
        print("⚠️ STATUS: Bad Request (400). Meistens ein Problem mit dem Modell-Namen oder der Region.")
    elif "API_KEY_INVALID" in error_msg:
        print("🔑 STATUS: Invalid API Key. Der Key in deiner .env stimmt nicht.")
    else:
        print(f"Oooops: {error_msg}")