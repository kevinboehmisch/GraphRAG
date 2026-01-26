import os
from dotenv import load_dotenv

# WICHTIG: Das sind neue Imports für die "Magie"
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph

load_dotenv()

# 1. Verbindung zur Datenbank (Diesmal über den LangChain Wrapper)
# Der Wrapper 'Neo4jGraph' macht das Speichern viel einfacher als der rohe Driver.
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 2. Das LLM Setup (Das Gehirn, das den Text analysiert)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0 # Temperatur 0, damit es keine Fakten erfindet
)

# 3. Der Transformer (Der "Bauarbeiter")
# Er nutzt das LLM, um Strukturen zu erkennen.
llm_transformer = LLMGraphTransformer(llm=llm)

# 4. Unser Input-Text (Simulieren wir ein Wikipedia-Snippet)
raw_text = """
Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. 
She was the first woman to win a Nobel Prize. 
She was married to Pierre Curie. 
In 1903, the Royal Swedish Academy of Sciences awarded Pierre Curie, Marie Curie, and Henri Becquerel the Nobel Prize in Physics.
SpaceX was founded by Elon Musk in 2002. It is located in Texas.
"""

# LangChain erwartet den Text in einem "Document"-Objekt
documents = [Document(page_content=raw_text)]

print("🤖 1. Gemini liest den Text und extrahiert Entitäten...")

# 5. DIE MAGIE: Text -> Graph Dokumente
# Das LLM entscheidet jetzt selbst: "Marie Curie ist eine Person", "Nobel Prize ist ein Award".
graph_documents = llm_transformer.convert_to_graph_documents(documents)

print(f"✅ Gefunden: {len(graph_documents[0].nodes)} Knoten und {len(graph_documents[0].relationships)} Beziehungen.")

# 6. Vorschau (Debugging - Damit du siehst, was passiert)
print("\n--- VORSCHAU WAS GESPEICHERT WIRD ---")
for node in graph_documents[0].nodes:
    # node.id = Der Name (z.B. "Marie Curie")
    # node.type = Der Typ (z.B. "Person")
    print(f"🔵 KNOTEN: {node.id} ({node.type})")

for rel in graph_documents[0].relationships:
    print(f"🔴 REL: {rel.source.id} --[{rel.type}]--> {rel.target.id}")
print("---------------------------------------\n")

# 7. Ab in die Datenbank damit!
user_input = input("Soll das wirklich in die DB gespeichert werden? (j/n): ")

if user_input.lower() == "j":
    print("💾 Speichere in Neo4j...")
    graph.add_graph_documents(graph_documents)
    print("🎉 Fertig! Geh jetzt auf http://localhost:7474 und gib ein: MATCH (n) RETURN n")
else:
    print("❌ Abgebrochen. Nichts gespeichert.")