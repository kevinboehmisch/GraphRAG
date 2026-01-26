import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

load_dotenv()
MODEL_NAME = "gemma3:27b"
FILE_PATH = "./data/long_data.txt" # Deine Datei mit der Attis Geschichte

print("🚀 START DEBUGGING: RAG vs. AGENT REASONING")
print("-" * 60)

llm = ChatOllama(model=MODEL_NAME, temperature=0)
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- TEIL 1: STANDARD RAG (Die "Dumb"-Variante) ---
print("\n🔍 ANALYSE STANDARD RAG (Was bekommt es wirklich?)")
if os.path.exists(FILE_PATH):
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    docs = loader.load()
    # Wir nehmen 1000 Token Chunks (fairer Standard)
    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(splits, embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Top 2 Chunks

    QUESTION = "Laut Prophezeiung droht Gefahr durch den 'Obsidian-Drachen'. Wer trägt diesen Gegenstand aktuell bei sich und wer ist der Vater dieser Person?"
    
    # 1. Wir holen die Chunks
    retrieved_docs = retriever.invoke(QUESTION)
    
    print(f"\n❓ FRAGE: {QUESTION}")
    print(f"\n📂 GEFUNDENE CHUNKS (Raw Input für das LLM):")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- CHUNK {i+1} ---")
        print(doc.page_content) # HIER: Der exakte 1:1 Text, den das RAG sieht
        print("----------------")

    # 2. Wir generieren die Antwort
    print("\n🤖 ANTWORT STANDARD RAG:")
    prompt_template = """Nutze NUR den folgenden Kontext für die Antwort:
    {context}
    
    Frage: {question}
    Antwort:"""
    
    # Wir bauen den Prompt manuell zusammen, damit du ihn siehst
    context_str = "\n\n".join([d.page_content for d in retrieved_docs])
    full_prompt = prompt_template.format(context=context_str, question=QUESTION)
    
    # Antwort generieren
    res = llm.invoke(full_prompt)
    print(res.content)

else:
    print("❌ Datei long_data.txt fehlt.")


# --- TEIL 2: GRAPH AGENT (Denkt er nach?) ---
print("\n" + "="*60)
print("🧠 ANALYSE GRAPH AGENT (Reasoning Trace)")
print("="*60)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

@tool
def search_entities(query: str):
    """Findet Knoten-Namen im Graphen."""
    try:
        vec = embedder.embed_query(query)
        res = graph.query("""
        CALL db.index.vector.queryNodes('entity_index', 3, $vec)
        YIELD node, score
        RETURN node.id, labels(node), score
        """, params={"vec": vec})
        return str(res)
    except: return "Fehler"

@tool
def get_neighborhood(entity_id: str):
    """Zeigt Verbindungen eines Knotens."""
    # Case-insensitive Suche
    res = graph.query("""
    MATCH (n) WHERE n.id =~ ('(?i)' + $id) OR n.name =~ ('(?i)' + $id)
    MATCH (n)-[r]-(m)
    RETURN n.id + ' --[' + type(r) + ']-- ' + m.id AS relation
    LIMIT 50
    """, params={"id": entity_id})
    return str(res)

tools = [search_entities, get_neighborhood]

# Das Template zwingt den Agenten zum "lauten Nachdenken" (Thought/Action/Observation)
template = """Du bist ein Detektiv.
Tools: {tools}
Tool Namen: {tool_names}

Deine Aufgabe: Finde die Antwort im Graphen durch logisches Schließen.

Format:
Thought: [Was überlege ich gerade?]
Action: [Welches Tool nutze ich?]
Action Input: [Was gebe ich ein?]
Observation: [Was kommt zurück?]
... (Wiederholen bis zur Lösung)
Final Answer: [Die Antwort]

Frage: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)

# WICHTIG: verbose=True zeigt dir das 'Denken' im Terminal
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # <--- DAS HIER ZEIGT DIR DEN GEDANKENPROZESS
    handle_parsing_errors=True
)

print(f"❓ FRAGE AN AGENT: {QUESTION}\n")
agent_executor.invoke({"input": QUESTION})