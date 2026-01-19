import os
from langchain_community.document_loaders import PyPDFLoader

# Sicherstellen, dass der Pfad stimmt
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "data", "demo.pdf")

print(f"🔍 Prüfe Datei: {pdf_path}")

if not os.path.exists(pdf_path):
    print("❌ FEHLER: Datei existiert nicht am angegebenen Pfad!")
else:
    print("✅ Datei gefunden. Versuche zu lesen...")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"📄 Anzahl Seiten: {len(pages)}")
        
        if len(pages) > 0:
            content = pages[0].page_content
            print(f"--- INHALT SEITE 1 (Erste 100 Zeichen) ---")
            print(f"'{content[:100]}'")
            print("------------------------------------------")
            
            if not content.strip():
                print("⚠️ WARNUNG: Seite 1 ist leer! Das PDF enthält wahrscheinlich nur Bilder.")
        else:
            print("⚠️ WARNUNG: Keine Seiten geladen.")
            
    except Exception as e:
        print(f"❌ CRASH: {e}")