"""
GraphRAG Builder für den Krimi-Workshop
========================================
MIT VOLLEM TERMINAL OUTPUT für Debugging!

Jede Entity und jede Beziehung wird sofort ausgegeben.
"""

import os
import re
import sys
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.graphs.graph_document import GraphDocument, Node as LCNode, Relationship as LCRel
from langchain_core.documents import Document

load_dotenv()

# =============================================================================
# KONFIGURATION
# =============================================================================

MODEL_NAME = "gemma3:12b"
EMBEDDING_MODEL = "nomic-embed-text"
CASE_FILE_PATH = "data/graphrag_workshop_case.md"

# Schema für den Krimi
ALLOWED_NODES = ["Person", "Ort", "Objekt", "Zeitpunkt", "Rolle"]
ALLOWED_RELATIONSHIPS = [
    "BEFINDET_SICH_IN",
    "BEWEGT",
    "SIEHT",
    "INTERAGIERT_MIT",
    "GEHOERT",
    "IST",              # "Praktikant" IST "Torben-Malte" / Aliase
    "HAT_ROLLE",        # Person HAT_ROLLE "CTO"
    "ARBEITET_ALS",     # Person ARBEITET_ALS Rolle
]

# =============================================================================
# LOGGING HELPER
# =============================================================================

def log(msg: str, indent: int = 0):
    """Sofortiger Print mit Flush."""
    prefix = "   " * indent
    print(f"{prefix}{msg}", flush=True)

def log_entity(entity_type: str, entity_id: str, section: str):
    """Loggt eine gefundene Entity."""
    print(f"      ✅ ENTITY: [{entity_type}] \"{entity_id}\"", flush=True)

def log_relationship(source: str, rel_type: str, target: str, time: str = ""):
    """Loggt eine gefundene Beziehung."""
    time_str = f" @{time}" if time else ""
    print(f"      🔗 REL: \"{source}\" --[{rel_type}]--> \"{target}\"{time_str}", flush=True)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ExtractedNode(BaseModel):
    id: str = Field(description="Eindeutiger Name der Entität")
    type: str = Field(description="Typ: Person, Ort, Objekt, oder Zeitpunkt")

class NodeList(BaseModel):
    nodes: List[ExtractedNode] = Field(default_factory=list)

class ExtractedRelationship(BaseModel):
    source: str = Field(description="Name des Startknotens")
    target: str = Field(description="Name des Zielknotens") 
    relation_type: str = Field(description="Art der Beziehung")
    time: str = Field(default="", description="Zeitpunkt falls bekannt")

class RelationshipList(BaseModel):
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

# =============================================================================
# TEXT CHUNKING
# =============================================================================

def split_into_sections(text: str) -> List[Dict[str, str]]:
    """Teilt den Text in logische Abschnitte."""
    sections = []
    
    # Split bei ### Headers (Zeugen, Verhöre)
    pattern = r'(###[^\n]*\n)'
    parts = re.split(pattern, text)
    
    current_title = "Einführung"
    current_content = ""
    
    for part in parts:
        if part.startswith("###"):
            # Vorherigen Abschnitt speichern
            if current_content.strip() and len(current_content) > 200:
                sections.append({
                    "title": current_title.strip(),
                    "content": current_content.strip()
                })
            current_title = part.replace("###", "").strip()
            current_content = ""
        else:
            current_content += part
    
    # Letzten Abschnitt
    if current_content.strip() and len(current_content) > 200:
        sections.append({
            "title": current_title.strip(),
            "content": current_content.strip()
        })
    
    # Fallback wenn keine ### gefunden
    if len(sections) == 0:
        # Split bei ## Headers
        pattern2 = r'(##[^\n]*\n)'
        parts = re.split(pattern2, text)
        
        for i, part in enumerate(parts):
            if len(part.strip()) > 300:
                sections.append({
                    "title": f"Abschnitt_{i}",
                    "content": part.strip()[:5000]
                })
    
    return sections

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_nodes(llm, text: str, section_title: str) -> List[ExtractedNode]:
    """Extrahiert Knoten Typ für Typ."""
    
    all_nodes = []
    parser = PydanticOutputParser(pydantic_object=NodeList)
    
    template = """Du extrahierst Entitäten aus einer Zeugenaussage.

ABSCHNITT: {section_title}

TEXT:
\"\"\"
{text}
\"\"\"

AUFGABE: Extrahiere ALLE Entitäten vom Typ "{node_type}".

{type_rules}

WICHTIG:
- NUR Typ "{node_type}" extrahieren
- Exakte Namen aus dem Text verwenden
- Keine Duplikate

{format_instructions}
"""

    type_rules = {
        "Person": """REGELN für Person:
- Vollständige Namen wenn möglich (z.B. "Peter Klein" nicht nur "Peter")
- Titel inkludieren (z.B. "Dr. Justus Vormann")
- Beispiele: Zeugen, Verdächtige, Opfer""",
        
        "Ort": """REGELN für Ort:
- Raumnummern (z.B. "Raum 404", "Raum 303")
- Bezeichnungen (z.B. "Teeküche", "Kopierraum", "Hauptkorridor")
- Gebäudeteile (z.B. "Etage 4", "Empfangsbereich")""",
        
        "Objekt": """REGELN für Objekt:
- Gegenstände (z.B. "goldene Trophäe", "roter USB-Stick")
- Getränke (z.B. "Chai-Latte", "Double Espresso")
- Dokumente (z.B. "veganes Druckerpapier")""",
        
        "Zeitpunkt": """REGELN für Zeitpunkt:
- NUR Uhrzeiten im Format "HH:MM" (z.B. "13:45", "14:00")
- Jede erwähnte Uhrzeit ist ein Zeitpunkt""",
        
        "Rolle": """REGELN für Rolle:
- Berufsbezeichnungen (z.B. "CTO", "Praktikant", "Sicherheitschef")
- Funktionen (z.B. "Reinigungskraft", "Head of Sales")
- NICHT die Person selbst, nur die Rolle/Funktion"""
    }

    prompt = PromptTemplate(
        template=template,
        input_variables=["text", "node_type", "section_title", "type_rules"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser

    for node_type in ALLOWED_NODES:
        log(f"🔍 Suche [{node_type}]...", indent=2)
        
        try:
            result = chain.invoke({
                "text": text[:4000],  # Max 4000 Zeichen
                "node_type": node_type,
                "section_title": section_title,
                "type_rules": type_rules.get(node_type, "")
            })
            
            # SOFORT jede Entity loggen!
            for node in result.nodes:
                node.type = node_type  # Typ erzwingen
                log_entity(node_type, node.id, section_title)
                all_nodes.append(node)
            
            if len(result.nodes) == 0:
                log(f"   (keine gefunden)", indent=2)
                
        except Exception as e:
            log(f"❌ FEHLER bei {node_type}: {str(e)[:100]}", indent=2)
    
    return all_nodes


def extract_relationships(llm, text: str, nodes: List[ExtractedNode], section_title: str) -> List[ExtractedRelationship]:
    """Extrahiert Beziehungen zwischen gefundenen Knoten."""
    
    if len(nodes) < 2:
        log("⏭️ Zu wenige Knoten für Beziehungen", indent=2)
        return []
    
    parser = PydanticOutputParser(pydantic_object=RelationshipList)
    
    # Knoten-Liste formatieren
    nodes_by_type = {}
    for n in nodes:
        if n.type not in nodes_by_type:
            nodes_by_type[n.type] = []
        nodes_by_type[n.type].append(n.id)
    
    node_list_str = ""
    for typ, ids in nodes_by_type.items():
        node_list_str += f"\n{typ}: {', '.join(ids)}"
    
    template = """Du analysierst Beziehungen in einer Zeugenaussage.

ABSCHNITT: {section_title}

TEXT:
\"\"\"
{text}
\"\"\"

GEFUNDENE ENTITÄTEN:
{node_list}

AUFGABE: Finde Beziehungen zwischen diesen Entitäten.

ERLAUBTE BEZIEHUNGSTYPEN:
- BEFINDET_SICH_IN: Person oder Objekt ist an einem Ort
- BEWEGT: Person nimmt/trägt/transportiert ein Objekt
- SIEHT: Person beobachtet eine andere Person oder ein Objekt
- INTERAGIERT_MIT: Person spricht mit / streitet mit einer anderen Person
- GEHOERT: Objekt gehört einer Person
- IST: Wenn zwei Bezeichnungen dieselbe Entität meinen (z.B. "Der Praktikant" IST "Torben-Malte")
- HAT_ROLLE: Person hat eine bestimmte Rolle/Funktion (z.B. "Peter Klein" HAT_ROLLE "CTO")

REGELN:
1. NUR die oben gelisteten Entitäten als source/target!
2. NUR die erlaubten Beziehungstypen!
3. Wenn eine Uhrzeit erwähnt wird, gib sie im "time" Feld an (Format: "HH:MM")
4. Erfinde KEINE Beziehungen die nicht im Text stehen!

{format_instructions}
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text", "node_list", "section_title"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    log(f"🔗 Suche Beziehungen zwischen {len(nodes)} Knoten...", indent=2)
    
    try:
        result = chain.invoke({
            "text": text[:4000],
            "node_list": node_list_str,
            "section_title": section_title
        })
        
        # SOFORT jede Beziehung loggen!
        for rel in result.relationships:
            log_relationship(rel.source, rel.relation_type, rel.target, rel.time)
        
        if len(result.relationships) == 0:
            log(f"   (keine gefunden)", indent=2)
            
        return result.relationships
        
    except Exception as e:
        log(f"❌ FEHLER bei Beziehungen: {str(e)[:100]}", indent=2)
        return []

# =============================================================================
# ENTITY MERGING
# =============================================================================

def normalize_id(entity_id: str) -> str:
    """
    Normalisiert Entity-Namen - NUR technisch, KEINE semantischen Aliase!
    
    Semantische Verbindungen wie "Praktikant" = "Torben-Malte" müssen
    durch Beziehungen im Graph erkannt werden, nicht hier!
    """
    normalized = entity_id.strip()
    
    # Nur technische Normalisierung:
    # 1. Whitespace normalisieren
    normalized = " ".join(normalized.split())
    
    # 2. Lowercase für Vergleich
    normalized = normalized.lower()
    
    # 3. Anführungszeichen entfernen
    normalized = normalized.replace('"', '').replace("'", "")
    
    return normalized


def merge_entities(all_nodes: List[ExtractedNode]) -> List[ExtractedNode]:
    """Führt Duplikate zusammen."""
    seen = {}
    
    for node in all_nodes:
        norm_id = normalize_id(node.id)
        if norm_id not in seen:
            seen[norm_id] = node
    
    return list(seen.values())

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_graph(file_path: str):
    """Hauptfunktion."""
    
    print("\n" + "="*70, flush=True)
    print("🕵️  KRIMI GRAPH BUILDER - VERBOSE MODE", flush=True)
    print("="*70, flush=True)
    
    # --- 1. Setup ---
    log("\n📌 SETUP...")
    
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    log("   Lösche alte Daten...")
    graph.query("MATCH (n) DETACH DELETE n")
    try:
        graph.query("DROP INDEX entity_index")
    except:
        pass
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    log(f"   Modell: {MODEL_NAME}")
    
    # --- 2. Lade Text ---
    log(f"\n📂 LADE DATEI: {file_path}")
    
    if not os.path.exists(file_path):
        log(f"❌ FEHLER: Datei nicht gefunden: {file_path}")
        sys.exit(1)
    
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    log(f"   {len(full_text)} Zeichen geladen")
    
    # --- 3. Chunking ---
    log("\n✂️  CHUNKING...")
    sections = split_into_sections(full_text)
    log(f"   {len(sections)} Abschnitte gefunden:")
    for i, s in enumerate(sections):
        log(f"   [{i+1}] {s['title'][:50]} ({len(s['content'])} Zeichen)")
    
    # --- 4. Extraktion ---
    print("\n" + "="*70, flush=True)
    print("🔬 STARTE EXTRAKTION", flush=True)
    print("="*70, flush=True)
    
    all_nodes: List[ExtractedNode] = []
    all_rels: List[ExtractedRelationship] = []
    
    for i, section in enumerate(sections):
        print(f"\n{'─'*70}", flush=True)
        print(f"📄 ABSCHNITT [{i+1}/{len(sections)}]: {section['title']}", flush=True)
        print(f"{'─'*70}", flush=True)
        
        if len(section['content']) < 100:
            log("⏭️ Übersprungen (zu kurz)")
            continue
        
        # Knoten extrahieren
        log("\n   === KNOTEN ===")
        nodes = extract_nodes(llm, section['content'], section['title'])
        all_nodes.extend(nodes)
        
        # Beziehungen extrahieren
        log("\n   === BEZIEHUNGEN ===")
        rels = extract_relationships(llm, section['content'], nodes, section['title'])
        all_rels.extend(rels)
        
        # Zwischenstand
        log(f"\n   📊 Zwischenstand: {len(all_nodes)} Knoten, {len(all_rels)} Beziehungen total")
    
    # --- 5. Merging ---
    print("\n" + "="*70, flush=True)
    print("🔀 ENTITY MERGING", flush=True)
    print("="*70, flush=True)
    
    merged_nodes = merge_entities(all_nodes)
    log(f"   Vorher: {len(all_nodes)} Knoten")
    log(f"   Nachher: {len(merged_nodes)} unique Knoten")
    
    # Alle finalen Entities auflisten
    log("\n   FINALE ENTITIES:")
    for n in merged_nodes:
        log(f"      [{n.type}] {n.id}")
    
    # Node-Map für Validierung
    valid_ids = {normalize_id(n.id) for n in merged_nodes}
    id_map = {normalize_id(n.id): n.id for n in merged_nodes}
    
    # Relationships validieren
    valid_rels = []
    log("\n   VALIDIERE BEZIEHUNGEN:")
    for rel in all_rels:
        src_norm = normalize_id(rel.source)
        tgt_norm = normalize_id(rel.target)
        
        if src_norm in valid_ids and tgt_norm in valid_ids:
            rel.source = id_map[src_norm]
            rel.target = id_map[tgt_norm]
            valid_rels.append(rel)
            log(f"      ✅ {rel.source} --[{rel.relation_type}]--> {rel.target}")
        else:
            log(f"      ❌ INVALID: {rel.source} --[{rel.relation_type}]--> {rel.target}")
    
    log(f"\n   {len(valid_rels)} von {len(all_rels)} Beziehungen valide")
    
    # --- 6. Graph Document ---
    print("\n" + "="*70, flush=True)
    print("🏗️  ERSTELLE GRAPH DOCUMENT", flush=True)
    print("="*70, flush=True)
    
    lc_nodes = [LCNode(id=n.id, type=n.type) for n in merged_nodes]
    node_map = {n.id: n for n in lc_nodes}
    
    lc_rels = []
    for rel in valid_rels:
        if rel.source in node_map and rel.target in node_map:
            lc_rel = LCRel(
                source=node_map[rel.source],
                target=node_map[rel.target],
                type=rel.relation_type
            )
            if rel.time:
                lc_rel.properties = {"time": rel.time}
            lc_rels.append(lc_rel)
    
    graph_doc = GraphDocument(
        nodes=lc_nodes,
        relationships=lc_rels,
        source=Document(page_content="Krimi Case File")
    )
    
    # --- 7. Embeddings ---
    print("\n" + "="*70, flush=True)
    print("⚡ BERECHNE EMBEDDINGS", flush=True)
    print("="*70, flush=True)
    
    for node in graph_doc.nodes:
        log(f"   Embedding für: {node.id}...")
        node.properties["name"] = node.id
        node.properties["embedding"] = embedder.embed_query(node.id)
    
    # --- 8. Speichern ---
    print("\n" + "="*70, flush=True)
    print("💾 SPEICHERE IN NEO4J", flush=True)
    print("="*70, flush=True)
    
    graph.add_graph_documents([graph_doc])
    log("   Graph Documents gespeichert")
    
    # Index
    graph.query("MATCH (n) WHERE n.embedding IS NOT NULL SET n:Entity")
    
    if graph_doc.nodes:
        vector_dim = len(graph_doc.nodes[0].properties["embedding"])
        graph.query(f"""
            CREATE VECTOR INDEX entity_index IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {vector_dim},
              `vector.similarity_function`: 'cosine'
            }}}}
        """)
        log(f"   Vector Index erstellt (dim={vector_dim})")
    
    # --- 9. FINALE ZUSAMMENFASSUNG ---
    print("\n" + "="*70, flush=True)
    print("🎉 FERTIG! ZUSAMMENFASSUNG", flush=True)
    print("="*70, flush=True)
    
    print(f"\n📊 STATISTIK:")
    print(f"   Knoten total: {len(lc_nodes)}")
    print(f"   Beziehungen total: {len(lc_rels)}")
    
    print(f"\n📋 ALLE KNOTEN:")
    type_counts = {}
    for n in lc_nodes:
        type_counts[n.type] = type_counts.get(n.type, 0) + 1
        print(f"   [{n.type}] {n.id}")
    
    print(f"\n📈 KNOTEN NACH TYP:")
    for t, c in sorted(type_counts.items()):
        print(f"   {t}: {c}")
    
    print(f"\n🔗 ALLE BEZIEHUNGEN:")
    for rel in lc_rels:
        time_str = f" @{rel.properties.get('time', '')}" if rel.properties else ""
        print(f"   {rel.source.id} --[{rel.type}]--> {rel.target.id}{time_str}")
    
    print("\n" + "="*70, flush=True)
    print("✅ GRAPH ERFOLGREICH ERSTELLT", flush=True)
    print("="*70 + "\n", flush=True)
    
    return graph_doc


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    build_graph(CASE_FILE_PATH)