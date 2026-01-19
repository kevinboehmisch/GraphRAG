import os
import time
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

# WICHTIG: Die lokalen Helden importieren
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

# --- 1. CONFIG & SETUP ---
print("\n🔌 1. Verbinde zur Datenbank...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# WICHTIG: Vor dem Test machen wir Tabula Rasa!
print("   🧹 Lösche alte Daten für sauberen Test...")
graph.query("MATCH (n) DETACH DELETE n")
# Wir löschen auch den Index, damit wir ihn sauber neu bauen können (falls nötig)
try:
    graph.query("DROP INDEX entity_index")
    print("   🗑️  Alter Index entfernt.")
except:
    pass # Index gab es wohl noch nicht

print("\n🦙 2. Lade lokale Modelle (Llama 3.1 & Nomic)...")
llm = ChatOllama(
    model="llama3:8b", 
    temperature=0  # Präzision ist wichtig
)

embedder = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Schema Definition
allowed_nodes = ["Person", "Organization", "Location", "Award", "Movie"]
# Hier schauen wir, ob er RIVAL kapiert
allowed_rels = ["FOUNDED", "LOCATED_IN", "MARRIED_TO", "WON", "ACTED_IN", "DIRECTED", "RIVAL"] 

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_rels
)

# --- 3. DER TEST-TEXT ---
raw_text = """
Steve Jobs founded Apple in California. 
He was a rival of Bill Gates, who founded Microsoft.
Apple is a technology company located in Cupertino.
"""
documents = [Document(page_content=raw_text)]
print(f"\n📄 3. Analysiere Text:\n   '{raw_text.strip().replace(chr(10), ' ')}'")

# --- 4. EXTRAKTION ---
print("\n🤖 4. Llama extrahiert Graphen (bitte warten)...")
start_time = time.time()
graph_documents = llm_transformer.convert_to_graph_documents(documents)
end_time = time.time()
print(f"   ✅ Fertig in {end_time - start_time:.2f} Sekunden.")

# --- 5. ANALYSE (VISUALISIERUNG IM TERMINAL) ---
print("\n🔍 5. VORSCHAU - Was hat Llama gefunden?")
nodes = graph_documents[0].nodes
rels = graph_documents[0].relationships

print(f"   📍 KNOTEN ({len(nodes)} Stück):")
for node in nodes:
    print(f"      - [{node.type}] {node.id}")

print(f"   🔗 BEZIEHUNGEN ({len(rels)} Stück):")
for rel in rels:
    print(f"      - {rel.source.id} --[{rel.type}]--> {rel.target.id}")

# --- 6. EMBEDDINGS & CLEANING ---
print("\n⚡ 6. Berechne Nomic-Embeddings...")

for node in nodes:
    # A) Name fixen
    if "name" not in node.properties:
        node.properties["name"] = node.id
    
    # B) Embedding berechnen
    # Wir messen die Zeit für einen Vektor, um die Speed zu checken
    t0 = time.time()
    vector = embedder.embed_query(node.id)
    t1 = time.time()
    
    # Check: Wie lang ist der Vektor? (Sollte 768 sein bei Nomic v1.5)
    dim = len(vector)
    preview = vector[:3] # Zeige die ersten 3 Zahlen
    print(f"   🧬 Vektor für '{node.id}': Dim={dim} | Zeit={t1-t0:.3f}s | Preview={preview}...")
    
    node.properties["embedding"] = vector

# --- 7. SPEICHERN ---
print("\n💾 7. Speichere in Neo4j...")
graph.add_graph_documents(graph_documents)

# --- 8. INDEXIERUNG ---
print("🏷️  8. Erstelle Label & Index...")
graph.query("MATCH (n) WHERE n.embedding IS NOT NULL SET n:Entity")

# Wir erstellen den Index direkt hier neu, passend zur Dimension von Nomic
vector_dim = len(nodes[0].properties["embedding"])
print(f"   ⚙️ Erstelle Vektor-Index für Dimension {vector_dim}...")

graph.query(f"""
    CREATE VECTOR INDEX entity_index IF NOT EXISTS
    FOR (n:Entity) ON (n.embedding)
    OPTIONS {{indexConfig: {{
     `vector.dimensions`: {vector_dim},
     `vector.similarity_function`: 'cosine'
    }}}}
""")

print("\n🎉 TEST ERFOLGREICH! Das Setup funktioniert.")