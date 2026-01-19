import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Lädt User/Passwort aus deiner .env Datei
load_dotenv()

# Verbindung herstellen
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def create_my_custom_db():
    with driver.session() as session:
        print("💥 1. LÖSCHE ALTE DATEN (Clean Slate)...")
        session.run("MATCH (n) DETACH DELETE n")
        session.run("DROP INDEX moviePlots IF EXISTS")
        
        print("🏗️ 2. ERSTELLE DEINE EIGENEN DATEN...")
        
        # Hier ist der Clou:
        # Wir erstellen Daten, wo der TEXT (Plot) ähnlich ist, aber das RATING und GENRE entscheidet!
        
        cypher_create = """
        // 1. Der Gewinner-Film (Action, Toys, Space, 5 Sterne)
        CREATE (m1:Movie {
            title: 'Galaxy Toys: The Awakening', 
            plot: 'A group of brave toys travel to other planets to fight alien invaders and save the universe.'
        })
        CREATE (g1:Genre {name: 'Action'})
        CREATE (a1:Person {name: 'Action Jackson'})
        CREATE (m1)-[:IN_GENRE]->(g1)
        CREATE (m1)<-[:ACTED_IN]-(a1)
        CREATE (u1:User {name: 'Fanboy'})
        CREATE (u1)-[:RATED {rating: 5.0}]->(m1)

        // 2. Der Verlierer-Film (Drama, Toys, Space, 1 Stern)
        CREATE (m2:Movie {
            title: 'The Sad Lonely Robot', 
            plot: 'A lonely toy robot travels to other planets looking for a friend but finds nothing.'
        })
        CREATE (g2:Genre {name: 'Drama'})
        CREATE (m2)-[:IN_GENRE]->(g2)
        CREATE (u1)-[:RATED {rating: 1.0}]->(m2)

        // 3. Der "Ablenkungs"-Film (Space, kein Toy, Action, mittelmäßig)
        CREATE (m3:Movie {
            title: 'Mars Rocks', 
            plot: 'A documentary about travelling to other planets and looking at rocks.'
        })
        CREATE (g3:Genre {name: 'Documentary'})
        CREATE (m3)-[:IN_GENRE]->(g3)
        CREATE (u1)-[:RATED {rating: 3.0}]->(m3)
        """
        session.run(cypher_create)
        print("   -> 3 Filme mit Genres und Ratings erstellt.")

        print("📐 3. ERSTELLE INDEX (768 Dim)...")
        session.run("""
            CREATE VECTOR INDEX moviePlots IF NOT EXISTS
            FOR (m:Movie) ON (m.plotEmbedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        time.sleep(3) # Kurz warten

        print("🧠 4. BERECHNE EMBEDDINGS...")
        movies = session.run("MATCH (m:Movie) RETURN m.title, m.plot").data()
        
        for movie in movies:
            title = movie['m.title']
            plot = movie['m.plot']
            print(f"   -> Embedding für: {title}")
            
            vector = embedder.embed_query(plot)
            
            session.run("""
                MATCH (m:Movie {title: $title})
                SET m.plotEmbedding = $vector
            """, title=title, vector=vector)

        print("\n✅ FERTIG! Deine Datenbank ist bereit für GraphRAG.")

if __name__ == "__main__":
    create_my_custom_db()
    driver.close()