import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Verbindung aufbauen
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# 2. Google Embeddings (text-embedding-004)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def fix_db():
    with driver.session() as session:
        print("1. Lösche alten OpenAI Index (1536 dimensions)...")
        try:
            session.run("DROP INDEX moviePlots IF EXISTS")
        except Exception as e:
            print(f"Info (Index existiert vllt nicht): {e}")

        print("2. Erstelle neuen Google Index (768 dimensions)...")
        # Wir erstellen einen neuen Vektor-Index, der 768 Dimensionen akzeptiert
        try:
            session.run("""
                CREATE VECTOR INDEX moviePlots IF NOT EXISTS
                FOR (m:Movie) ON (m.plotEmbedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
        except Exception as e:
            print(f"Info (Index Erstellung): {e}")
        
        print("3. Hole 20 Filme zum Aktualisieren...")
        result = session.run("MATCH (m:Movie) WHERE m.plot IS NOT NULL RETURN m.title, m.plot LIMIT 20")
        movies = [record.data() for record in result]

        print(f"4. Berechne neue Vektoren für {len(movies)} Filme...")
        
        for movie in movies:
            title = movie['m.title']
            plot = movie['m.plot']
            
            try:
                # Embedding berechnen
                vector = embeddings.embed_query(plot)
                
                # --- HIER WAR DER FEHLER ---
                # Korrektur: Wir fügen 'YIELD node' und 'RETURN count(*)' hinzu
                session.run("""
                    MATCH (m:Movie {title: $title})
                    CALL db.create.setVectorProperty(m, 'plotEmbedding', $vector)
                    YIELD node
                    RETURN count(*)
                """, title=title, vector=vector)
                
                print(f" -> Updated: {title}")
                
                # Pause für Free Tier
                time.sleep(2) 
                
            except Exception as e:
                print(f"Fehler bei '{title}': {e}")

    print("\nFERTIG! Jetzt kannst du vector_retriever.py starten.")

if __name__ == "__main__":
    fix_db()
    driver.close()