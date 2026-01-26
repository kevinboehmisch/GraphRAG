import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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

# Aufräumen
graph.query("MATCH (n) DETACH DELETE n")
try:
    graph.query("DROP INDEX entity_index")
except:
    pass

# Modell Setup - Temperature 0 ist PFLICHT für Logik!
llm = ChatOllama(model="gemma3:27b", temperature=0) # oder llama3.1
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- 2. DATENSTRUKTUREN (Pydantic) ---
# Wir definieren strikte Klassen, damit das LLM genau weiß, wie der Output aussehen muss.

class IdentifiedNode(BaseModel):
    id: str = Field(description="Der Name der Entität, z.B. 'Apple' oder 'Steve Jobs'")
    type: str = Field(description="Der Typ der Entität")

class IdentifiedNodeList(BaseModel):
    nodes: List[IdentifiedNode]

class IdentifiedRelationship(BaseModel):
    source: str = Field(description="Name des Startknotens")
    target: str = Field(description="Name des Zielknotens")
    type: str = Field(description="Art der Beziehung, z.B. FOUNDED, LOCATED_IN")

class RelationshipList(BaseModel):
    relationships: List[IdentifiedRelationship]

# --- 3. DER "GUIDED" EXTRAKTOR ---

def extract_nodes_step_by_step(text: str, allowed_nodes: List[str]) -> List[IdentifiedNode]:
    """Iteriert durch jeden Knotentyp einzeln."""
    all_nodes = []
    
    parser = PydanticOutputParser(pydantic_object=IdentifiedNodeList)
    
    # Der Prompt konzentriert sich NUR auf EINEN Typ
    template = """Du bist ein Experte für Daten-Extraktion.
    Deine Aufgabe: Extrahiere ALLE Entitäten vom Typ '{node_type}' aus dem folgenden Text.
    
    Text: "{text}"
    
    WICHTIG:
    - Extrahiere NUR '{node_type}'. Ignoriere alles andere.
    - Gib das Ergebnis strikt als JSON zurück.
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["node_type", "text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser

    for node_type in allowed_nodes:
        print(f"   👉 Suche nach Typ: [{node_type}]...")
        try:
            result = chain.invoke({"node_type": node_type, "text": text})
            count = len(result.nodes)
            print(f"      ✅ {count} gefunden: {[n.id for n in result.nodes]}")
            # Typ sicherstellen
            for n in result.nodes:
                n.type = node_type # Überschreiben zur Sicherheit
                all_nodes.append(n)
        except Exception as e:
            print(f"      ❌ Fehler bei {node_type}: {e}")
            
    return all_nodes

def extract_relationships_guided(text: str, nodes: List[IdentifiedNode], allowed_rels: List[str]) -> List[IdentifiedRelationship]:
    """Sucht Beziehungen NUR zwischen den bereits gefundenen Knoten."""
    
    parser = PydanticOutputParser(pydantic_object=RelationshipList)
    
    node_list_str = ", ".join([f"{n.id} ({n.type})" for n in nodes])
    
    template = """Du bist ein Experte für Knowledge Graphs.
    
    Text: "{text}"
    
    Bereits erkannte Entitäten (Knoten):
    {node_list}
    
    Deine Aufgabe:
    Finde Beziehungen zwischen diesen Entitäten basierend auf dem Text.
    Erlaubte Beziehungstypen: {allowed_rels}
    
    REGELN:
    - Nutze NUR die oben gelisteten Entitäten als Source und Target. Erfinde keine neuen Knoten!
    - Nutze NUR die erlaubten Beziehungstypen.
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text", "node_list", "allowed_rels"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    print(f"   👉 Suche nach Beziehungen zwischen {len(nodes)} Knoten...")
    try:
        result = chain.invoke({
            "text": text, 
            "node_list": node_list_str,
            "allowed_rels": ", ".join(allowed_rels)
        })
        print(f"      ✅ {len(result.relationships)} Beziehungen gefunden.")
        return result.relationships
    except Exception as e:
        print(f"      ❌ Fehler bei Beziehungen: {e}")
        return []

# --- 4. HAUPTPROGRAMM ---

# --- 3. DER "MEATY" TEST-TEXT ---
raw_text = """
NVIDIA is a technology giant headquartered in Santa Clara. 
The company is led by CEO Jensen Huang. 
NVIDIA designs the H100 GPU, which is critical for artificial intelligence. 
However, the H100 is not manufactured in the USA. 
It is manufactured by TSMC, a semiconductor foundry located in Taiwan. 
Microsoft is a major customer that uses the H100 for its Azure cloud services. 
Microsoft is based in Redmond and competes with Google in the AI sector.
Google also designs its own chips called TPUs, but they rely on TSMC for manufacturing as well.
"""

# Wir passen die Listen an das Szenario an
target_nodes = ["Organization", "Person", "Location", "Product"]

target_rels = [
    "HEADQUARTERED_IN", # Wo sitzt die Firma?
    "LED_BY",           # Wer ist Chef?
    "DESIGNS",          # Wer erfindet das Produkt?
    "MANUFACTURED_BY",  # Wer baut es wirklich?
    "LOCATED_IN",       # Wo steht die Fabrik?
    "USES",             # Wer kauft es?
    "COMPETES_WITH"     # Wer sind die Rivalen?
]
extracted_nodes = extract_nodes_step_by_step(raw_text, target_nodes)
extracted_rels = extract_relationships_guided(raw_text, extracted_nodes, target_rels)

# --- 5. KONSTRUKTION DES GRAPH DOCUMENTS ---
print("\n🏗️  5. Baue GraphDocument...")

# Konvertierung in LangChain Objekte
lc_nodes = [LangChainNode(id=n.id, type=n.type) for n in extracted_nodes]
lc_rels = []

# Hilfs-Mapping um Duplikate zu vermeiden
node_map = {n.id: n for n in lc_nodes}

for rel in extracted_rels:
    # Sicherstellen, dass Source und Target existieren
    if rel.source in node_map and rel.target in node_map:
        lc_rels.append(LangChainRel(
            source=node_map[rel.source],
            target=node_map[rel.target],
            type=rel.type
        ))
    else:
        print(f"   ⚠️ Ignoriere Beziehung {rel.source}->{rel.target}, da Knoten fehlen.")

graph_doc = GraphDocument(nodes=lc_nodes, relationships=lc_rels, source=Document(page_content=raw_text))

# --- 6. EMBEDDINGS & SAVE ---
print("\n⚡ 6. Berechne Embeddings & Speichere...")

for node in graph_doc.nodes:
    # Wichtig: Embedding immer einzeln berechnen
    node.properties["name"] = node.id
    node.properties["embedding"] = embedder.embed_query(node.id)

graph.add_graph_documents([graph_doc])

# Index erstellen
graph.query("MATCH (n) WHERE n.embedding IS NOT NULL SET n:Entity")
vector_dim = len(graph_doc.nodes[0].properties["embedding"])
graph.query(f"""
    CREATE VECTOR INDEX entity_index IF NOT EXISTS
    FOR (n:Entity) ON (n.embedding)
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {vector_dim},
      `vector.similarity_function`: 'cosine'
    }}}}
""")

print("\n🎉 FERTIG! Schau in Neo4j nach.")
print(f"   Gefundene Knoten: {[n.id for n in lc_nodes]}")
print(f"   Gefundene Kanten: {[f'{r.source.id}->{r.type}->{r.target.id}' for r in lc_rels]}")