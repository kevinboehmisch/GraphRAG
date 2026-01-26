#1. Text lesen & Graphen extrahieren.
#2. Daten bereinigen (Namen fixen).
#3. Sofort Embeddings berechnen.
#4. Speichern.
#5. Label :Entity vergeben (damit der Index greift).

import os
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph

load_dotenv()

# --- SETUP ---
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0 
)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Schema-Definition (Damit kein Quatsch reinkommt)
allowed_nodes = ["Person", "Organization", "Location", "Award", "Movie"]
allowed_rels = ["FOUNDED", "LOCATED_IN", "MARRIED_TO", "WON", "ACTED_IN", "DIRECTED", "RIVAL"]

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_rels
)

# --- INPUT TEXT ---
raw_text = """
Steve Jobs founded Apple in California. 
He was a rival of Bill Gates, who founded Microsoft.
Apple is a technology company located in Cupertino.
"""
documents = [Document(page_content=raw_text)]

# --- PROZESS ---
print("🤖 1. Extrahiere Graphen aus Text...")
graph_documents = llm_transformer.convert_to_graph_documents(documents)

print(f"⚡ 2. Berechne Vektoren für {len(graph_documents[0].nodes)} Knoten...")

# Wir gehen durch jeden extrahierten Knoten im Speicher
for node in graph_documents[0].nodes:
    # A) Name fixen (wie besprochen)
    if "name" not in node.properties:
        node.properties["name"] = node.id
    
    # B) EMBEDDING BERECHNEN (Hier passiert die Magie!)
    # Wir nehmen die ID (den Namen) als Basis für den Vektor
    vector = embedder.embed_query(node.id)
    
    # C) Vektor direkt in den Knoten schreiben
    node.properties["embedding"] = vector

print("💾 3. Speichere alles in Neo4j...")
graph.add_graph_documents(graph_documents)

# --- NACHBEHANDLUNG (WICHTIG!) ---
# Damit unser Vektor-Index (der auf :Entity hört) die neuen Knoten findet,
# müssen wir ihnen allen den Stempel :Entity aufdrücken.
print("🏷️  4. Verteile Label ':Entity'...")
graph.query("""
    MATCH (n) 
    WHERE n.embedding IS NOT NULL 
    SET n:Entity
""")

print("🎉 Fertig! Daten sind drin, Vektoren sind drin, Index ist happy.")