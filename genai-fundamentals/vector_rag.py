import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- 1. DER DEBUG-ADAPTER ---
# Dieser Teil macht das "Unsichtbare" sichtbar
class GoogleGenAIAdapter:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input, *args, **kwargs):
        # Google-Fix: Störenden Parameter entfernen
        if "system_instruction" in kwargs:
            kwargs.pop("system_instruction")
        
        # --- DEBUGGING START ---
        print("\n" + "="*60)
        print("🕵️  WAS PASSIERT IM HINTERGRUND? (LLM INPUT)")
        print("="*60)
        
        # Wir zerlegen die Nachricht, die an Google geht
        if isinstance(input, list):
            for msg in input:
                if isinstance(msg, SystemMessage):
                    print(f"\n[SYSTEM ANWEISUNG]:\n{msg.content.strip()}")
                elif isinstance(msg, HumanMessage):
                    print(f"\n[DATEN AUS DEINER DB + USER FRAGE]:\n{msg.content.strip()}")
        else:
            print(f"[RAW INPUT]: {input}")
            
        print("="*60 + "\n")
        # --- DEBUGGING ENDE ---

        # Weiterleitung an Google
        return self.llm.invoke(input, *args, **kwargs)

# 2. Setup
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# 3. Embedder (768 Dimensionen)
embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 4. Retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# 5. LLM Setup
google_llm_internal = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7 # 0 = Sei präzise, erfinde nichts
)

llm = GoogleGenAIAdapter(google_llm_internal)

# 6. Pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# 7. Ausführung
query_text = "Find me movies about toys coming alive and explain strictly based on the context why they match."

print(f"Start Anfrage: '{query_text}'...\n")

try:
    # Wir holen die Top 3 Ergebnisse aus der DB
    response = rag.search(
        query_text=query_text, 
        retriever_config={"top_k": 3}
    )
    
    print("\n📝 FERTIGE ANTWORT (Generiert aus den Daten oben):")
    print("-" * 50)
    print(response.answer)
    print("-" * 50)

except Exception as e:
    print(f"\n❌ Fehler: {e}")

driver.close()