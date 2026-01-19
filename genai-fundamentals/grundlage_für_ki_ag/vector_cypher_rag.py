import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

# --- 1. CONFIG & ADAPTER ---
class GoogleGenAIAdapter:
    def __init__(self, llm):
        self.llm = llm
    def invoke(self, input, *args, **kwargs):
        if "system_instruction" in kwargs:
            kwargs.pop("system_instruction")
        return self.llm.invoke(input, *args, **kwargs)

# Verbindung
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Modelle
embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
llm_adapter = GoogleGenAIAdapter(llm)

# --- 2. INDEX ERSTELLEN (WICHTIG!) ---
# Da wir neue Daten (Marie Curie) haben, brauchen wir einen Vektor-Index auf diesen Daten.
# Wir erstellen einen Index namens "entity_index" auf ALLEN Knoten, die eine 'id' haben.

INDEX_NAME = "entity_index"

def create_vector_index():
    print("⚙️ Prüfe/Erstelle Vektor-Index...")
    with driver.session() as session:
        # Prüfen ob Index existiert
        result = session.run("SHOW VECTOR INDEXES WHERE name = $name", name=INDEX_NAME)
        if result.peek():
            print("   Index existiert bereits. Überspringe Erstellung.")
            return

        # Index erstellen (Wir betten die 'id' Eigenschaft ein, da dort der Name steht)
        # Hinweis: In einer echten App würde man dies einmalig machen, nicht bei jedem Start.
        print("   Erstelle neuen Index... (Das dauert kurz)")
        session.run("""
            CREATE VECTOR INDEX entity_index IF NOT EXISTS
            FOR (n:Person | Organization | Location | Award) ON (n.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 768,
             `vector.similarity_function`: 'cosine'
            }}
        """)
        
        # Embeddings berechnen und speichern!
        # Das hier füllt die Datenbank mit Vektoren für deine neuen Knoten.
        print("   Berechne Embeddings für existierende Knoten...")
        nodes = session.run("MATCH (n) WHERE n.id IS NOT NULL AND n.embedding IS NULL RETURN n.id AS text, elementId(n) AS id")
        
        updates = []
        for record in nodes:
            vector = embedder.embed_query(record["text"])
            updates.append({"id": record["id"], "embedding": vector})
            
        if updates:
            session.run("""
            UNWIND $updates AS update
            MATCH (n) WHERE elementId(n) = update.id
            SET n.embedding = update.embedding
            """, updates=updates)
            print(f"   ✅ {len(updates)} Knoten vektorisiert!")
        else:
            print("   Keine neuen Knoten zum Vektorisieren gefunden.")

# Einmal ausführen
create_vector_index()


# --- 3. DIE "CONTEXT" QUERY ---
# DAS ist der Teil, der GraphRAG mächtig macht.
# 1. Finde den Knoten per Vektor (node)
# 2. Hole ALLE direkten Nachbarn (neighbor)
# 3. Gib alles als Text zurück an das LLM

retrieval_query = """
MATCH (node) // Das ist der Treffer aus der Vektor-Suche (z.B. Marie Curie)
RETURN
    "Main Entity: " + node.id AS entity,
    
    // Hier passiert die Magie: Wir sammeln alle Beziehungen ein
    collect {
        MATCH (node)-[r]-(neighbor)
        RETURN startNode(r).id + " --[" + type(r) + "]--> " + endNode(r).id
    } AS context

// Sortieren nach Ähnlichkeit (score kommt vom Vector Retriever automatisch)
ORDER BY score DESC
"""

# --- 4. Retriever & RAG Setup ---
retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX_NAME,
    embedder=embedder,
    retrieval_query=retrieval_query,
)

rag = GraphRAG(retriever=retriever, llm=llm_adapter)

# --- 5. Abfrage ---
# Teste Fragen zu deinen importierten Daten
questions = [
    "Who is married to Marie Curie and what did they win?",
    "Where is SpaceX located and who founded it?"
]

print("\n🔎 Starte Context-Suche...\n")

for q in questions:
    print(f"❓ FRAGE: {q}")
    response = rag.search(q, return_context=True)
    
    # Zeige, was der Graph gefunden hat (den Context)
    if response.retriever_result.items:
        print("💡 GEFUNDENER CONTEXT (Aus dem Graphen):")
        first_result = response.retriever_result.items[0] # Erster Treffer
        # Das content Feld enthält was wir im Cypher RETURN definiert haben
        print(f"   {first_result.content}") 
    else:
        print("   (Kein Context gefunden)")
        
    print(f"\n📝 ANTWORT: {response.answer}")
    print("-" * 60)

driver.close()