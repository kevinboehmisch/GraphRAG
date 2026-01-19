import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- 1. DER ALLES-ANZEIGEN ADAPTER ---
class GoogleGenAIAdapter:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input, *args, **kwargs):
        if "system_instruction" in kwargs:
            kwargs.pop("system_instruction")
        
        print("\n" + "█"*60)
        print("🕵️  ALLES WAS AN GEMINI GEHT (UNGEFILTERT)")
        print("█"*60)
        
        # Wir iterieren durch ALLES, egal was für ein Nachrichtentyp es ist
        if isinstance(input, list):
            for i, msg in enumerate(input):
                type_name = type(msg).__name__
                content = msg.content
                print(f"\n--- NACHRICHT {i+1} ({type_name}) ---")
                print(content)
        else:
            print(f"[RAW STRING INPUT]: {input}")
            
        print("█"*60 + "\n")
        
        return self.llm.invoke(input, *args, **kwargs)

# 2. Setup
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 3. DIE QUERY (ROBUSTER GEMACHT) ---
# WICHTIG: Wir nutzen OPTIONAL MATCH.
# Warum? Falls ein Film KEINE Ratings oder KEINE Schauspieler hat,
# würde ein normales MATCH den Film komplett wegwerfen.
retrieval_query = """
MATCH (node)
RETURN 
  node.title AS title, 
  node.plot AS plot, 
  score AS similarityScore, 
  
  collect { 
    OPTIONAL MATCH (node)-[:IN_GENRE]->(g) 
    RETURN g.name 
  } as genres, 
  
  collect { 
    OPTIONAL MATCH (node)<-[:ACTED_IN]->(a) 
    RETURN a.name 
  } as actors, 
  
  collect {
    OPTIONAL MATCH (node)<-[r:RATED]-() 
    RETURN r.rating
  } as ratings

ORDER BY size(ratings) DESC
"""

# 4. Retriever
retriever = VectorCypherRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

# 5. LLM
google_llm_internal = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

llm = GoogleGenAIAdapter(google_llm_internal)

# 6. Pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# 7. Suche
query_text = "Find me a movie about toys that people really hated (low rating)."
print(f"🔎 Suche nach: '{query_text}'...\n")

try:
    # return_context=True zeigt uns, ob der Retriever überhaupt Daten findet
    response = rag.search(
        query_text=query_text, 
        retriever_config={"top_k": 3},
        return_context=True
    )
    
    # Check: Hat der Retriever was gefunden?
    if not response.retriever_result.items:
        print("❌ ALARM: Der Retriever hat KEINE Daten aus der DB geholt.")
        print("   Mögliche Ursache: Keine 'RATED' oder 'ACTED_IN' Beziehungen im Graphen?")
    else:
        print(f"✅ Der Retriever hat {len(response.retriever_result.items)} Datensätze gefunden.")

    print("\n📝 ANTWORT:")
    print("-" * 50)
    print(response.answer)
    print("-" * 50)

except Exception as e:
    print(f"\n❌ Fehler: {e}")

driver.close()