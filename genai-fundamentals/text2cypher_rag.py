import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

load_dotenv()

# --- 1. Adapter (Damit Gemini mit Neo4j-GraphRAG spricht) ---
class GoogleGenAIAdapter:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input, *args, **kwargs):
        # Google mag "system_instruction" als Argument hier oft nicht, wir fangen es ab
        if "system_instruction" in kwargs:
            kwargs.pop("system_instruction")
            
        response = self.llm.invoke(input, *args, **kwargs)
        # Text2Cypher erwartet oft einen String, LangChain gibt eine Message.
        # Wir geben die Message zurück, die Library extrahiert meist .content selbst.
        return response

# 2. Verbindung
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# 3. LLM Setup
google_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # Nimm ein intelligentes Modell für Code-Generierung!
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0 # WICHTIG: Code muss präzise sein, nicht kreativ.
)
llm_adapter = GoogleGenAIAdapter(google_llm)

# 4. Das Schema (Die Landkarte für die KI)
# Wir beschreiben genau, was in DEINER 'setup_custom_db.py' erstellt wurde.
neo4j_schema = """
Node properties:
Movie {title: STRING, plot: STRING}
Genre {name: STRING}
Person {name: STRING}
User {name: STRING}

Relationship properties:
RATED {rating: FLOAT}

The relationships:
(:Movie)-[:IN_GENRE]->(:Genre)
(:Movie)<-[:ACTED_IN]-(:Person)
(:Movie)<-[:RATED]-(:User)
"""

# 5. Few-Shot Examples (Training)
# Wir zeigen der KI, wie man Cypher schreibt.
examples = [
    "USER INPUT: 'Which movies represent the Action genre?' QUERY: MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Action'}) RETURN m.title",
    "USER INPUT: 'Who acted in Galaxy Toys?' QUERY: MATCH (m:Movie {title: 'Galaxy Toys: The Awakening'})<-[:ACTED_IN]-(p:Person) RETURN p.name"
]

# 6. Der Retriever (Der Übersetzer)
retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm_adapter,
    neo4j_schema=neo4j_schema,
    examples=examples,
)

# 7. Pipeline
rag = GraphRAG(retriever=retriever, llm=llm_adapter)

# --- TEST-FRAGEN FÜR DEINE DB ---
# Wir fragen Dinge, die man nur durch Rechnen/Zählen im Graphen lösen kann.
questions = [
    "How many movies are in the database?",
    "What is the average rating of Galaxy Toys?",
    "List all actors in the movie Galaxy Toys.",
    "Which movie has the worst rating?"
]

print("🤖 Starte Text2Cypher Test...\n")

for q in questions:
    print(f"❓ FRAGE: {q}")
    try:
        response = rag.search(query_text=q, return_context=True)
        
        # HIER SIEHST DU DEN GENERIERTEN CODE:
        generated_cypher = response.retriever_result.metadata["cypher"]
        
        print(f"💻 GENERIERTER CYPHER CODE:\n   {generated_cypher}")
        print(f"📝 ANTWORT: {response.answer}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Fehler bei '{q}': {e}")
        print("-" * 50)

driver.close()