import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Setup
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def reset_and_fill():
    with driver.session() as session:
        print("🔍 ANALYSE VORHER:")
        # Mal schauen, was da ist
        try:
            cnt = session.run("MATCH (m:Movie) WHERE m.plotEmbedding IS NOT NULL RETURN count(m) as c").single()['c']
            print(f"   Filme mit Vektoren: {cnt}")
        except:
            print("   (Konnte Anzahl nicht lesen, mache weiter...)")

        print("\n🧹 AUFRÄUMEN (Lösche Index & alte Vektoren)...")
        # 1. Index weg
        try:
            session.run("DROP INDEX moviePlots IF EXISTS")
            print("   -> Index 'moviePlots' gelöscht.")
        except Exception as e:
            print(f"   -> Info: {e}")

        # 2. Vektoren entfernen
        session.run("MATCH (m:Movie) REMOVE m.plotEmbedding")
        print("   -> Alte Vektoren aus Filmen entfernt.")

        print("\n🏗️ NEUAUFBAU (Google Index 768 Dimensionen)...")
        # 3. Index neu erstellen
        try:
            session.run("""
                CREATE VECTOR INDEX moviePlots IF NOT EXISTS
                FOR (m:Movie) ON (m.plotEmbedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            print("   -> Neuer Index 'moviePlots' (768 Dim) erstellt.")
        except Exception as e:
            print(f"❌ FEHLER beim Index erstellen: {e}")
            return

        print("   -> Warte 5 Sekunden auf Index-Initialisierung...")
        time.sleep(5)

        print("\n📥 DATEN LADEN (Hole 20 Filme)...")
        movies = session.run("MATCH (m:Movie) WHERE m.plot IS NOT NULL RETURN m.title, m.plot LIMIT 20").data()
        
        print(f"   -> Berechne Embeddings für {len(movies)} Filme mit Google...")
        
        count = 0
        for movie in movies:
            title = movie['m.title']
            plot = movie['m.plot']
            
            try:
                # Vektor holen
                vector = embedder.embed_query(plot)
                
                # Check beim ersten Film
                if count == 0:
                    print(f"   ℹ️ Vektor-Check: {len(vector)} Dimensionen. (Soll: 768) -> {'OK' if len(vector)==768 else 'FEHLER'}")

                # --- HIER IST DER FIX ---
                # Statt CALL nutzen wir einfach SET. Das ist robuster.
                session.run("""
                    MATCH (m:Movie {title: $title})
                    SET m.plotEmbedding = $vector
                """, title=title, vector=vector)
                
                print(f"   -> Embedded: {title}")
                count += 1
                time.sleep(1.0) # Rate Limit Schutz
                
            except Exception as e:
                print(f"   ❌ Fehler bei '{title}': {e}")

        print(f"\n✅ FERTIG! {count} Filme erfolgreich neu indexiert.")

if __name__ == "__main__":
    reset_and_fill()
    driver.close()