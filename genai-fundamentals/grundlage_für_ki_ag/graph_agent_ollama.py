import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)
llm = ChatOllama(model="llama3:8b", temperature=0)
embedder = OllamaEmbeddings(model="nomic-embed-text")

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
    clean_id = entity_id.strip().strip('"').strip("'").split('\n')[0]
    query = """
    MATCH (n)
    WHERE n.id =~ ('(?i)' + $entity_id) OR n.name =~ ('(?i)' + $entity_id)
    MATCH (n)-[r]-(neighbor)
    RETURN n.id + ' --[' + type(r) + ']--> ' + neighbor.id AS relationship
    LIMIT 15
    """
    results = graph.query(query, params={"entity_id": clean_id})
    return results if results else f"Keine Nachbarn für '{clean_id}' gefunden."

tools = [search_entities, get_neighborhood]
# --- DER STRIKTE PROMPT (JETZT MIT ALLEN VARIABLEN) ---
template = """Du bist ein präziser Graph-Detektiv. Deine Aufgabe ist es, einen Pfad von A nach B im Graphen zu beweisen.

STRIKTE ARBEITSWEISE:
1. Nutze 'search_entities' EINMALIG, um den Start-Knoten (z.B. SpaceX) zu finden.
2. Nutze DANACH NUR NOCH 'get_neighborhood', um von einem Knoten zum nächsten zu springen.
3. Wenn du den Sitz (Stadt) gefunden hast, MUSST du 'get_neighborhood' auf diese Stadt anwenden, um die dort hergestellten Raketen zu finden.
4. Erfinde NIEMALS Verbindungen. Wenn 'get_neighborhood' keine Kante zeigt, existiert sie nicht!

TOOLS:
------
Dir stehen folgende Werkzeuge zur Verfügung:
{tools}

FORMAT (STRENG EINHALTEN):
--------------------------
Thought: Was muss ich als nächstes tun?
Action: Das Tool, das du nutzt (muss eines von [{tool_names}] sein)
Action Input: Die Eingabe für das Tool
Observation: Das Ergebnis des Tools

... (Wiederhole Thought/Action/Observation so oft wie nötig)

Thought: Ich habe den Pfad im Graphen eindeutig gesehen.
Final Answer:
Antwort: [Die Rakete]
Beweis: [Start-Knoten] --[Beziehung]--> [Stadt] --[Beziehung]--> [Rakete]

Frage: {input}
{agent_scratchpad}"""

# WICHTIG: LangChain braucht genau diese Variablen im Template!
prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

query = "Welche Rakete wird in der Stadt hergestellt, in der SpaceX seinen Sitz hat?"
print(f"\n🕵️ Untersuchung startet...\n")
agent_executor.invoke({"input": query})