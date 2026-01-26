import os
import re
from langchain_community.document_loaders import PyPDFLoader

# --- KONFIGURATION ---
INPUT_FILE = "./data/guide.pdf"       # Wie heißt deine PDF?
OUTPUT_FILE = "long_data.txt"  # So soll die Textdatei heißen

# Ab welchen Wörtern soll abgeschnitten werden? (Müll am Ende)
# WICHTIG: Prüfe, wie das in deinem PDF genau heißt (oft "References", "Bibliography" etc.)
STOP_MARKERS = [
    "\nReferences", 
    "\nBibliography", 
    "\nIndex", 
    "\nGlossary"
]

def clean_content(text):
    # 1. Seitenzahlen entfernen (Zeilen, die nur eine Zahl enthalten)
    # Regex: Sucht nach Zeilen, die nur Ziffern und Whitespace haben
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # 2. Kopfzeilen/Fußzeilen entfernen (Optional, wenn sie immer gleich sind)
    # Beispiel: "Ultimate Guide to LLMs" auf jeder Seite entfernen
    # text = text.replace("Ultimate Guide to LLMs", "")
    
    # 3. Referenzen [1], [12] entfernen (stören Vektorsuche oft)
    text = re.sub(r'\[\d+\]', '', text)
    
    # 4. Zu viele Leerzeilen reduzieren
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text

def convert_pdf_to_clean_text():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Fehler: Datei '{INPUT_FILE}' nicht gefunden.")
        return

    print(f"📖 Lade PDF: {INPUT_FILE}...")
    loader = PyPDFLoader(INPUT_FILE)
    pages = loader.load()
    
    full_text = ""
    print(f"   -> {len(pages)} Seiten gefunden.")

    # Alle Seiten zusammenfügen
    for page in pages:
        full_text += page.page_content + "\n"

    # --- DER MAGIC CLEANER ---
    print("🧹 Reinige Text...")
    
    # A) Müll am Ende abschneiden
    cut_off_index = len(full_text)
    found_marker = None
    
    for marker in STOP_MARKERS:
        # Wir suchen das LETZTE Vorkommen oder das erste in einer Überschrift
        idx = full_text.find(marker)
        if idx != -1 and idx < cut_off_index:
            # Sicherheitscheck: Marker muss weit hinten sein (nicht im Inhaltsverzeichnis am Anfang)
            if idx > len(full_text) * 0.5: 
                cut_off_index = idx
                found_marker = marker

    if found_marker:
        print(f"   ✂️  Schneide Text ab bei '{found_marker.strip()}'...")
        full_text = full_text[:cut_off_index]
    else:
        print("   ⚠️  Kein Abbruch-Marker (References etc.) gefunden. Text bleibt komplett.")

    # B) Formatierung bereinigen
    cleaned_text = clean_content(full_text)

    # Speichern
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    print(f"✅ Fertig! Gespeichert als '{OUTPUT_FILE}'.")
    print(f"   Original Länge: {len(full_text)} Zeichen")
    print(f"   Neue Länge:     {len(cleaned_text)} Zeichen")

if __name__ == "__main__":
    convert_pdf_to_clean_text()