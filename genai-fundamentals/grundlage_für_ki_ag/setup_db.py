import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Konfiguration
INDEX_NAME = "entity_index"
DIMENSION = 768 # Passend für Nomic Embed Text v1.5 (und Google)

def setup_database():
    print("🧨 Starte TOTAL-RESET der Datenbank...")
    
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

    with driver.session() as session:
        # 1. ALLES LÖSCHEN (Der wichtigste Teil!)
        print("   🧹 Lösche alle Knoten und Beziehungen...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # 2. Alten Index löschen (falls er existiert, um sauber neu zu starten)
        print("   🗑️  Lösche alten Index (falls vorhanden)...")
        try:
            session.run(f"DROP INDEX {INDEX_NAME}")
        except Exception:
            pass # Gab wohl noch keinen, egal.

        # 3. Index NEU erstellen
        print(f"   ⚙️  Erstelle neuen Vektor-Index '{INDEX_NAME}'...")
        session.run(f"""
            CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {{indexConfig: {{
             `vector.dimensions`: {DIMENSION},
             `vector.similarity_function`: 'cosine'
            }}}}
        """)
        
        # 4. Constraints erstellen
        print("   🛡️  Erstelle Constraints...")
        session.run("CREATE CONSTRAINT unique_person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT unique_org_id IF NOT EXISTS FOR (n:Organization) REQUIRE n.id IS UNIQUE")
        
    driver.close()
    print("✅ Datenbank ist jetzt KOMPLETT LEER und bereit.")

if __name__ == "__main__":
    setup_database()