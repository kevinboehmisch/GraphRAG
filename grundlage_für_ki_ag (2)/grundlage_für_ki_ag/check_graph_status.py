import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings

load_dotenv()

print("🔍 GRAPH DIAGNOSE START")

# 1. Verbindung prüfen
try:
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    print("✅ Verbindung zur DB steht.")
except Exception as e:
    print(f"❌ Keine Verbindung: {e}")
    exit()

# 2. Knoten zählen
node_count = graph.query("MATCH (n) RETURN count(n) as c")[0]['c']
print(f"📊 Anzahl Knoten: {node_count}")

if node_count == 0:
    print("❌ ACHTUNG: Der Graph ist LEER! Du musst 'build_real_graph.py' nochmal laufen lassen.")
    exit()

# 3. Index prüfen
indexes = graph.query("SHOW INDEXES")
found_index = False
for idx in indexes:
    if idx['name'] == 'entity_index':
        found_index = True
        print(f"✅ Vektor-Index 'entity_index' gefunden (Status: {idx['state']}).")
        
if not found_index:
    print("❌ FEHLER: Kein Vektor-Index gefunden! Der Agent KANN nicht suchen.")
    # Versuch der Reparatur
    print("🛠️ Versuche Index zu reparieren...")
    try:
        graph.query("""
            CREATE VECTOR INDEX entity_index IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
        """)
        print("   -> Reparatur-Befehl gesendet. Warte kurz...")
    except Exception as e:
        print(f"   -> Reparatur gescheitert: {e}")

# 4. Suche testen (Der echte Grund für den "Fehler")
print("\n🧪 Teste Vektor-Suche manuell...")
embedder = OllamaEmbeddings(model="nomic-embed-text")
try:
    vec = embedder.embed_query("Drache")
    print("✅ Embedding generiert.")
    
    res = graph.query("""
    CALL db.index.vector.queryNodes('entity_index', 3, $vec)
    YIELD node, score
    RETURN node.id, score
    """, params={"vec": vec})
    
    print(f"✅ Suche erfolgreich! Ergebnis: {res}")
except Exception as e:
    print(f"❌ FEHLER BEI DER SUCHE: {e}")
    print("   -> Das ist der Grund, warum der Agent abstürzt.")

print("\nDIAGNOSE ENDE.")