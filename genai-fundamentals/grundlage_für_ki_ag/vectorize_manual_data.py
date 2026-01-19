import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)
embedder = OllamaEmbeddings(model="nomic-embed-text")

print("⚡ Berechne Vektoren für manuelle Daten...")

# Alle Knoten ohne Embedding holen
nodes = graph.query("MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN n.id AS id")
# Falls du vorher alles gelöscht hast, nimm alle:
nodes = graph.query("MATCH (n:Entity) RETURN n.id AS id")

for record in nodes:
    node_id = record['id']
    vector = embedder.embed_query(node_id)
    graph.query(
        "MATCH (n {id: $id}) SET n.embedding = $vector",
        params={"id": node_id, "vector": vector}
    )
    print(f"   ✅ {node_id} indexiert.")

print("🎉 Fertig! Dein Agent kann diese Daten jetzt finden.")