import os
import time
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- CONFIG ---
TXT_FILE = "data/demo.txt"

# 1. Verbindung zur Datenbank
print("🔌 Verbinde zur DB...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 2. KI-Modelle (Gemini)
print("✨ Lade Gemini Flash & Google Embeddings...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0 
)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 3. Schema (Englisch - mal sehen ob Gemini das übersetzen kann!)
allowed_nodes = ["Person", "Organization", "Location", "Award", "Movie", "Product", "Event", "Rocket"]
allowed_rels = ["FOUNDED", "LOCATED_IN", "MARRIED_TO", "WON", "ACTED_IN", "RIVAL", "PRODUCED", "LAUNCHED", "CEO_OF", "INVENTED", "DEVELOPED", "OWNS"]

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_rels
)

# --- ABLAUF ---

# A) Text laden
if not os.path.exists(TXT_FILE):
    print(f"❌ FEHLER: Datei '{TXT_FILE}' fehlt!")
    exit()

loader = TextLoader(TXT_FILE, encoding="utf-8")
raw_documents = loader.load()

# B) Chunking
# Wir bleiben bei 2000 für den fairen Vergleich, auch wenn Gemini mehr könnte.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Begrenzung für den Test
documents_to_process = documents[:6] # Die ersten 6 reichen für den Vergleich
print(f"✂️  Text zerteilt. Verarbeite {len(documents_to_process)} Chunks mit Gemini...")

# C) Extraktion mit DEBUG-AUSGABE
count_nodes = 0

for i, doc in enumerate(documents_to_process):
    print(f"\n==========================================")
    preview = doc.page_content[:50].replace('\n', ' ')
    print(f"🔄 Chunk {i+1}: '{preview}...'")
    
    try:
        # Extraktion
        result = llm_transformer.convert_to_graph_documents([doc])
        
        if not result:
            print("   ⚠️ (Leer)")
            time.sleep(2)
            continue
            
        nodes = result[0].nodes
        rels = result[0].relationships

        # --- HIER IST DIE AUSGABE FÜR DICH ---
        print(f"   📍 Gefundene Knoten ({len(nodes)}):")
        for n in nodes[:5]: # Zeige die ersten 5
             print(f"      - [{n.type}] {n.id}")
        if len(nodes) > 5: print("      ... (und weitere)")

        print(f"   🔗 Gefundene Beziehungen ({len(rels)}):")
        for r in rels:
            print(f"      - {r.source.id} --[{r.type}]--> {r.target.id}")
        # -------------------------------------

        # Embeddings & Speichern
        print(f"   ⚡ Berechne Vektoren...", end="", flush=True)
        for node in nodes:
            if "name" not in node.properties:
                node.properties["name"] = node.id
            # Embedding
            node.properties["embedding"] = embedder.embed_query(node.id)
        print(" Fertig.")

        graph.add_graph_documents(result)
        print("   💾 Gespeichert.")
        
        count_nodes += len(nodes)

        # WICHTIG: Bremse für Google Free Tier
        print("   ⏳ Cool-down (5s)...")
        time.sleep(5)
        
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        print("   Warte 30 Sekunden...")
        time.sleep(30)

# D) Index
print("\n🏷️  Verteile Label ':Entity'...")
graph.query("MATCH (n) WHERE n.embedding IS NOT NULL SET n:Entity")
print(f"🎉 Fertig! {count_nodes} Knoten mit Gemini importiert.")