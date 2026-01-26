import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

# --- KONFIGURATION ---
FILE_PATH = "data/long_data.txt"  
MODEL_NAME = "gemma3:27b"          
CHUNK_SIZE = 1000                  
CHUNK_OVERLAP = 150                # Etwas mehr Overlap, damit keine Beziehung an der Schnittstelle verloren geht

print(f"\n🚀 STARTE ATTIS GRAPH-BUILDER (MODERN PIPELINE)")
print(f"   Modell: {MODEL_NAME}")
print("-" * 60)

# --- 1. SETUP ---
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)
llm = ChatOllama(model=MODEL_NAME, temperature=0) # Temp 0 ist wichtig für JSON Stabilität
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- 2. CLEANUP ---
graph.query("MATCH (n) DETACH DELETE n")
try: graph.query("DROP INDEX entity_index")
except: pass

# --- 3. DATEN ---
if not os.path.exists(FILE_PATH):
    print("❌ Datei fehlt!")
    exit()

loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
splits = text_splitter.split_documents(documents)
print(f"📥 Verarbeite {len(splits)} Chunks (Das ist sicher für den Speicher).")

# --- 4. DER EXTRAKTOR (Das Herzstück) ---
# Wir definieren das Schema passend zu ATTIS
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[
        "Person", "God", "Weapon", "Prophecy", "Location", "Object"
    ],
    allowed_relationships=[
        "WIELDS", "FATHER_OF", "WARNS_OF", "LOCATED_IN", "MENTIONS", "FOUND_BY", "HAS_SYMBOL", "PART_OF"
    ],
    # OPTIONAL: Hier zwingen wir das Modell, gründlich zu sein
    strict_mode=False 
)

# --- 5. BUILD LOOP ---
print("\n🏗️  Starte Extraktion...")

for i, chunk in enumerate(splits):
    print(f"\n👉 CHUNK {i+1}/{len(splits)}")
    
    # Vorschau
    preview = chunk.page_content.replace('\n', ' ')[:80]
    print(f"   Inhalt: '{preview}...'")

    try:
        # Hier passiert Extraction (Nodes + Rels gleichzeitig)
        graph_docs = llm_transformer.convert_to_graph_documents([chunk])
        
        nodes = graph_docs[0].nodes
        rels = graph_docs[0].relationships
        
        # --- DEIN GEWÜNSCHTER OUTPUT ---
        if len(nodes) > 0:
            print(f"   ✅ {len(nodes)} Knoten | {len(rels)} Kanten")
            for n in nodes:
                print(f"      • ({n.type}) {n.id}")
            for r in rels:
                print(f"      • {r.source.id} -[{r.type}]-> {r.target.id}")
        else:
            print("   ⚪ (Leer - Wahrscheinlich Müll/Inventar)")

        graph.add_graph_documents(graph_docs)
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

# --- 6. EMBEDDINGS & INDEX ---
print("\n⚡ Indexierung...")
graph.query("MATCH (n) WHERE n.embedding IS NULL SET n.embedding = null") # Init
nodes = graph.query("MATCH (n) RETURN n.id AS id")
for record in nodes:
    nid = record['id']
    vec = embedder.embed_query(nid)
    graph.query("MATCH (n) WHERE n.id = $id SET n.embedding = $vec", params={"id": nid, "vec": vec})

graph.query("MATCH (n) SET n:Entity")
try:
    graph.query("""
        CREATE VECTOR INDEX entity_index IF NOT EXISTS
        FOR (n:Entity) ON (n.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """)
except: pass

print("\n🎉 FERTIG! Attis-Graph steht.")