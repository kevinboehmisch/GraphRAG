import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Verbindung aufbauen
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def check_status():
    with driver.session() as session:
        print("\n🏥 DATENBANK DIAGNOSE REPORT")
        print("=============================")

        # 1. Gibt es überhaupt Daten?
        total_nodes = session.run("MATCH (n) RETURN count(n) as c").single()['c']
        print(f"📦 Knoten gesamt:        {total_nodes}")

        if total_nodes == 0:
            print("❌ ERGEBNIS: Die Datenbank ist KOMPLETT LEER.")
            return

        # 2. Gibt es Filme?
        movie_count = session.run("MATCH (m:Movie) RETURN count(m) as c").single()['c']
        print(f"🎬 Filme (Nodes):        {movie_count}")

        # 3. Haben die Filme Vektoren (für die Suche)?
        vector_count = session.run("MATCH (m:Movie) WHERE m.plotEmbedding IS NOT NULL RETURN count(m) as c").single()['c']
        print(f"🔢 Filme mit Vektoren:   {vector_count}")
        
        if vector_count == 0:
            print("⚠️ WARNUNG: Filme sind da, aber der Vektor-Index ist leer! Suche wird nicht funktionieren.")

        # 4. Gibt es Beziehungen (Wichtig für dein zweites Skript)?
        # Wir suchen nach ACTED_IN (Schauspieler) und RATED (Bewertungen)
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']
        print(f"🔗 Beziehungen gesamt:   {rel_count}")

        acted_in = session.run("MATCH ()-[r:ACTED_IN]->() RETURN count(r) as c").single()['c']
        print(f"   - davon ACTED_IN:     {acted_in}")

        rated = session.run("MATCH ()-[r:RATED]->() RETURN count(r) as c").single()['c']
        print(f"   - davon RATED:        {rated}")

        print("=============================")
        
        # FAZIT
        if rel_count == 0:
            print("❌ DIAGNOSE FÜR FEHLER: Dein 'VectorCypherRetriever' stürzt ab, weil es KEINE Beziehungen gibt.")
            print("   Der Vektor findet zwar den Film, aber die Query findet keine Schauspieler/Ratings dazu.")
        elif vector_count == 0:
            print("❌ DIAGNOSE FÜR FEHLER: Dein 'VectorRetriever' findet nichts, weil keine Embeddings da sind.")
        else:
            print("✅ DIAGNOSE: Die Daten sehen eigentlich gut aus.")

if __name__ == "__main__":
    try:
        check_status()
    except Exception as e:
        print(f"Verbindungsfehler: {e}")
    finally:
        driver.close()