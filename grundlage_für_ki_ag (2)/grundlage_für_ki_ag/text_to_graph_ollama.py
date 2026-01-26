import os
import time
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

TXT_FILE = "data/demo.txt"

# 1. SETUP
print("🔌 Verbinde zur DB...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 2. MODELLE
print("🦙 Lade Llama 3.1 & Nomic...")
llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0)
embedder = OllamaEmbeddings(model="nomic-embed-text")

# 3. SCHEMA AUF DEUTSCH (WICHTIG!) 🇩🇪
# Wir erlauben dem Modell deutsche Begriffe, damit es den deutschen Text versteht.
allowed_nodes = ["Person", "Organisation", "Ort", "Auszeichnung", "Film", "Produkt", "Rakete", "Konzept"]
allowed_rels = [
    "GEGRUENDET", "SITZ_IN", "VERHEIRATET_MIT", "GEWONNEN", 
    "HAT_GEARBEITET_BEI", "RIVALE_VON", "HERGESTELLT", 
    "GESTARTET", "CEO_VON", "ERFUNDEN", "ENTWICKELT", "TEIL_VON"
]

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_rels,
    # Wir können versuchen, das Modell strikt zu instruieren (klappt bei Ollama manchmal gemischt)
    strict_mode=False 
)

# --- ABLAUF ---

if not os.path.exists(TXT_FILE):
    print(f"❌ Datei {TXT_FILE} fehlt!")
    exit()

loader = TextLoader(TXT_FILE, encoding="utf-8")
raw_documents = loader.load()

# B) Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=500)
documents = text_splitter.split_documents(raw_documents)

# LIMITIERUNG FÜR TEST (Wie gewünscht)
documents = documents[:15]

print(f"✂️  Verarbeite die ersten {len(documents)} Chunks...")

# C) Loop
count_nodes = 0
count_rels = 0 # Zähler für Beziehungen

for i, doc in enumerate(documents):
    
    # Kurze Vorschau
    preview = doc.page_content[:40].replace('\n', ' ')
    print(f"\n--- Chunk {i+1}: '{preview}...' ---")
    
    try:
        graph_docs = llm_transformer.convert_to_graph_documents([doc])
        
        if not graph_docs:
            print("   ⚠️  Nichts gefunden.")
            continue
            
        nodes = graph_docs[0].nodes
        rels = graph_docs[0].relationships
        
        print(f"   📍 Knoten ({len(nodes)}): {[n.id for n in nodes[:3]]}...") # Zeige nur erste 3 Namen
        
        # JETZT WIRD ES SPANNEND: Findet er Beziehungen?
        if len(rels) == 0:
             print("   ⚠️  KEINE BEZIEHUNGEN GEFUNDEN (Prüfe Prompt/Sprache)")
        else:
            print(f"   🔗 Beziehungen ({len(rels)}):")
            for r in rels:
                print(f"      - {r.source.id} --[{r.type}]--> {r.target.id}")

        # Embeddings & Speichern
        for node in nodes:
            if "name" not in node.properties:
                node.properties["name"] = node.id
            node.properties["embedding"] = embedder.embed_query(node.id)
        
        graph.add_graph_documents(graph_docs)
        print("   💾 Gespeichert.")
        
        count_nodes += len(nodes)
        count_rels += len(rels)
        
    except Exception as e:
        print(f" ❌ Fehler: {e}")

# D) Index
print("\n🏷️  Verteile Label ':Entity'...")
graph.query("MATCH (n) WHERE n.embedding IS NOT NULL SET n:Entity")

print(f"\n🎉 FERTIG! Knoten: {count_nodes} | Beziehungen: {count_rels}")
if count_rels == 0:
    print("❌ WARNUNG: Immer noch keine Beziehungen. Llama kommt mit dem Deutsch->Schema Mapping nicht klar.")