import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

# --- 1. CONFIG & VERBINDUNG ---
# Verbindung zur DB (Nur zum Lesen)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Embedder (Um DEINE FRAGE in Zahlen zu verwandeln)
embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# LLM (Um die Antwort zu schreiben)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Kleiner Helper für LangChain Kompatibilität
class GoogleGenAIAdapter:
    def __init__(self, llm):
        self.llm = llm
    def invoke(self, input, *args, **kwargs):
        if "system_instruction" in kwargs:
            kwargs.pop("system_instruction")
        return self.llm.invoke(input, *args, **kwargs)

llm_adapter = GoogleGenAIAdapter(llm)

# --- 2. RETRIEVAL STRATEGIE ---
# Wir nutzen den Index "entity_index", den setup_db.py erstellt hat.

INDEX_NAME = "entity_index"

# Die Query holt den gefundenen Knoten UND seine Nachbarn (Context)
retrieval_query = """
MATCH (node) 
RETURN
    "Main Entity: " + node.id AS entity,
    collect {
        MATCH (node)-[r]-(neighbor)
        RETURN startNode(r).id + " --[" + type(r) + "]--> " + endNode(r).id
    } AS context
"""

# --- 3. RAG INITIALISIEREN ---
retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX_NAME,
    embedder=embedder,        # Wichtig: Embeddet deine Frage live
    retrieval_query=retrieval_query,
)

rag = GraphRAG(retriever=retriever, llm=llm_adapter)

# --- 4. SUCHE ---
# Hier kannst du Fragen stellen
questions = [
    "Who founded Apple and where is it?", 
    "Who is the rival of Bill Gates?",
    "Tell me about Steve Jobs."
]

print("\n🔎 Starte Graph-Suche (Context Mode)...\n")

try:
    for q in questions:
        print(f"❓ FRAGE: {q}")
        response = rag.search(q, return_context=True)
        
        # Optional: Zeige, was der Graph im Hintergrund gefunden hat
        if response.retriever_result.items:
            print("💡 CONTEXT (aus DB):")
            # Wir nehmen nur die ersten 300 Zeichen des Contexts, damit die Konsole nicht explodiert
            context_text = str(response.retriever_result.items[0].content)
            print(f"   {context_text[:300]}...") 
        
        print(f"\n📝 ANTWORT: {response.answer}")
        print("-" * 60)

finally:
    driver.close()