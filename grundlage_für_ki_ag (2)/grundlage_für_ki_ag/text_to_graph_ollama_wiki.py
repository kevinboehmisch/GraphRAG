import os
import time
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# WICHTIG: Der Loader für Wikipedia
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.graphs.graph_document import GraphDocument, Node as LangChainNode, Relationship as LangChainRel
from langchain_core.documents import Document

load_dotenv()

# --- 1. CONFIG ---
print("\n🔌 1. Verbinde zur Datenbank...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Aufräumen (Clean Slate)
graph.query("MATCH (n) DETACH DELETE n")
try:
    graph.query("DROP INDEX entity_index")
except:
    pass

# Modell Setup (Gemma oder Llama)
llm = ChatOllama(model="gemma3:27b", temperature=0) 
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- 2. DATENSTRUKTUREN ---
class IdentifiedNode(BaseModel):
    id: str = Field(description="Name der Entität")
    type: str = Field(description="Typ der Entität")

class IdentifiedNodeList(BaseModel):
    nodes: List[IdentifiedNode]

class IdentifiedRelationship(BaseModel):
    source: str = Field(description="Startknoten")
    target: str = Field(description="Zielknoten")
    type: str = Field(description="Beziehungstyp (Englisch, UPPERCASE)")

class RelationshipList(BaseModel):
    relationships: List[IdentifiedRelationship]

# --- 3. EXTRAKTOR-FUNKTIONEN ---
def extract_nodes_step_by_step(text: str, allowed_nodes: List[str]) -> List[IdentifiedNode]:
    all_nodes = []
    parser = PydanticOutputParser(pydantic_object=IdentifiedNodeList)
    
    template = """Extrahiere ALLE Entitäten vom Typ '{node_type}' aus dem Text.
    Text Ausschnitt: "{text_snippet}..."
    
    Gib NUR JSON zurück.
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["node_type", "text_snippet"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser

    # Wir verarbeiten den Text in Chunks, falls er zu lang ist, 
    # aber für dieses Beispiel kürzen wir ihn einfach auf das Limit des Modells.
    safe_text = text[:6000] # Limit für lokale LLMs, damit sie nicht abstürzen

    for node_type in allowed_nodes:
        print(f"   👉 Suche nach Typ: [{node_type}]...")
        try:
            # Wir geben dem LLM nur die ersten 6000 Zeichen, das reicht für extrem viele Daten
            result = chain.invoke({"node_type": node_type, "text_snippet": safe_text})
            print(f"      ✅ {len(result.nodes)} gefunden.")
            for n in result.nodes:
                n.type = node_type
                all_nodes.append(n)
        except Exception as e:
            print(f"      ❌ Fehler bei {node_type}: {e}")
            
    return all_nodes

def extract_relationships_guided(text: str, nodes: List[IdentifiedNode], allowed_rels: List[str]) -> List[IdentifiedRelationship]:
    parser = PydanticOutputParser(pydantic_object=RelationshipList)
    node_list_str = ", ".join([f"{n.id}" for n in nodes])
    
    template = """Finde Beziehungen zwischen diesen Entitäten im Text.
    
    Text Ausschnitt: "{text_snippet}..."
    
    Entitäten: {node_list}
    Erlaubte Beziehungen: {allowed_rels}
    
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["text_snippet", "node_list", "allowed_rels"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    
    print(f"   👉 Suche Beziehungen...")
    try:
        # Auch hier: Safe Text Limit
        result = chain.invoke({
            "text_snippet": text[:6000], 
            "node_list": node_list_str,
            "allowed_rels": ", ".join(allowed_rels)
        })
        return result.relationships
    except Exception as e:
        print(f"      ❌ Fehler: {e}")
        return []

# --- 4. WIKIPEDIA LADEN (DAS IST NEU) ---

print("\n🌍 4. Lade 'PayPal Mafia' von Wikipedia...")
# Wir laden den Artikel über die PayPal Mafia. 
# Alternativen: "Elon Musk", "OpenAI", "Volkswagen Group"
search_query = "PayPal Mafia" 
docs = WikipediaLoader(query=search_query, load_max_docs=1, lang="en").load()

if not docs:
    print("❌ Nichts gefunden!")
    exit()

full_text = docs[0].page_content
print(f"    📄 Geladener Text: {len(full_text)} Zeichen.")
print(f"    📖 Titel: {docs[0].metadata['title']}")
print(f"    📝 Vorschau: {full_text[:200]}...")

# --- 5. HAUPTPROGRAMM ---

target_nodes = ["Person", "Organization", "Product"]
target_rels = ["FOUNDED", "WORKED_AT", "CREATED", "INVESTED_IN", "OWNED_BY"]

# A) Knoten
extracted_nodes = extract_nodes_step_by_step(full_text, target_nodes)

# B) Beziehungen
extracted_rels = extract_relationships_guided(full_text, extracted_nodes, target_rels)

# C) Graph bauen
print("\n🏗️  Baue GraphDocument...")
lc_nodes = [LangChainNode(id=n.id, type=n.type) for n in extracted_nodes]
lc_rels = []
node_map = {n.id: n for n in lc_nodes}

for rel in extracted_rels:
    if rel.source in node_map and rel.target in node_map:
        lc_rels.append(LangChainRel(
            source=node_map[rel.source],
            target=node_map[rel.target],
            type=rel.type
        ))

graph_doc = GraphDocument(nodes=lc_nodes, relationships=lc_rels, source=Document(page_content=full_text))

# D) Speichern & Vektorisieren
print("\n💾 Speichere in Neo4j & Vektorisierung...")
for node in graph_doc.nodes:
    node.properties["name"] = node.id
    # Kleiner Sleep, damit Ollama nicht überhitzt
    time.sleep(0.05) 
    node.properties["embedding"] = embedder.embed_query(node.id)

graph.add_graph_documents([graph_doc])

# Index neu erstellen
graph.query("MATCH (n) WHERE n.embedding IS NOT NULL SET n:Entity")
try:
    vector_dim = len(graph_doc.nodes[0].properties["embedding"])
    graph.query(f"""
        CREATE VECTOR INDEX entity_index IF NOT EXISTS
        FOR (n:Entity) ON (n.embedding)
        OPTIONS {{indexConfig: {{
        `vector.dimensions`: {vector_dim},
        `vector.similarity_function`: 'cosine'
        }}}}
    """)
except:
    pass

print("\n🎉 FERTIG! Dein Graph enthält jetzt echte Wikipedia-Daten.")
print(f"   Knoten: {len(lc_nodes)} | Kanten: {len(lc_rels)}")