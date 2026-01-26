import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- 1. SETUP ---
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Du nutzt Gemma 3 (27b) - das ist super für komplexe Logik!
llm = ChatOllama(model="gemma3:27b", temperature=0)
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- 2. TOOLS (Bleiben gleich, die sind super) ---
@tool
def search_entities(query: str):
    """NUR für den ersten Schritt! Findet den Namen des Start-Knotens."""
    clean_query = query.strip().strip('"').strip("'")
    vector_query = """
    CALL db.index.vector.queryNodes('entity_index', 3, $embedding)
    YIELD node, score
    RETURN coalesce(node.id, node.name) AS entity, labels(node) AS type
    """
    embedding = embedder.embed_query(clean_query)
    return graph.query(vector_query, params={"embedding": embedding})

@tool
def get_neighborhood(entity_id: str):
    """DIE EINZIGE QUELLE FÜR FAKTEN. Zeigt echte Verbindungen im Graphen."""
    # Cleaning für Robustheit
    clean_id = entity_id.strip().strip('"').strip("'").split('\n')[0]
    query = """
    MATCH (n)
    WHERE n.id =~ ('(?i)' + $entity_id) OR n.name =~ ('(?i)' + $entity_id)
    MATCH (n)-[r]-(neighbor)
    RETURN n.id + ' --[' + type(r) + ']--> ' + neighbor.id AS relationship
    LIMIT 20
    """
    results = graph.query(query, params={"entity_id": clean_id})
    return results if results else f"Keine Nachbarn für '{clean_id}' gefunden."

tools = [search_entities, get_neighborhood]

# --- 3. DER GENERISCHE PROMPT (Wichtigste Änderung!) ---
# Wir haben "Raketen" entfernt und durch "logische Kette" ersetzt.

template = """Du bist ein intelligenter Graph-Analyst für Lieferketten.
Deine Aufgabe ist es, Abhängigkeiten über mehrere Stationen hinweg zu finden.

STRIKTE REGELN:
1. Nutze 'search_entities' EINMALIG für den Start-Knoten (z.B. Microsoft).
2. Nutze DANACH 'get_neighborhood', um dich von Knoten zu Knoten zu hangeln.
3. Folge dem Pfad logisch:
   - Wer nutzt was? (USES)
   - Wer stellt das her? (MANUFACTURED_BY)
   - Wo ist der Hersteller? (LOCATED_IN)
4. Erfinde NIEMALS Verbindungen!

TOOLS:
------
{tools}

FORMAT:
-------
Thought: Welchen Knoten muss ich als nächstes untersuchen?
Action: [{tool_names}]
Action Input: [Name des Knotens]
Observation: [Ergebnis]

... (Wiederhole bis zum Ziel)

Thought: Ich habe die Kette bis zum Zielland verfolgt.
Final Answer:
Antwort: [Das Land/Der Ort]
Beweis: [Start] --[Rel]--> [Zwischenschritt] --[Rel]--> [Ziel]

Frage: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# --- 4. EXECUTION ---
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True, 
    max_iterations=15 # Chip-Ketten sind lang, gib ihm Zeit!
)

# --- 5. DIE NEUE FRAGE ---
# Das ist die ultimative Logik-Prüfung für deinen Graphen
# Neue Frage für den PayPal-Mafia Graphen
query = "Welche anderen Firmen hat der Besitzer von Twitter noch gegründet?"

print(f"\n🕵️ Untersuchung der Lieferkette startet...\n")
agent_executor.invoke({"input": query})