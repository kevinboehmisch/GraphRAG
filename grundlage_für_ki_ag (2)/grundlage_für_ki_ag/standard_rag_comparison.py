import sys
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- SETUP ---
# Nutze hier dein Modell
MODEL_NAME = "gemma3:27b" 

print(f"🚀 Starte 'Disjoint Data' Test mit {MODEL_NAME}...")

llm = ChatOllama(model=MODEL_NAME, temperature=0)
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- 1. DIE DATEN (Das Silo-Problem) ---
# Wir simulieren Datenbank-Einträge, die nichts voneinander wissen.

docs = [
    # DOKUMENT 1: Info über Twitter (aber KEIN Wort über SpaceX)
    Document(page_content="Twitter was acquired by Elon Musk in 2022. He serves as the CTO and owner of the social media platform."),
    
    # DOKUMENT 2: Info über SpaceX (aber KEIN Wort über Twitter)
    Document(page_content="SpaceX designs, manufactures and launches advanced rockets and spacecraft. The company was founded in 2002 by Elon Musk."),
    
    # NOISE (Ablenkung für die Vektorsuche) - Damit 'founder' nicht sofort zu Musk führt
    Document(page_content="Amazon was founded by Jeff Bezos in Bellevue, Washington."),
    Document(page_content="Microsoft was founded by Bill Gates and Paul Allen."),
    Document(page_content="Google was founded by Larry Page and Sergey Brin."),
    Document(page_content="Facebook's founder Mark Zuckerberg is a tech mogul."),
    Document(page_content="Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.")
]

print(f"📥 Datenbank geladen: {len(docs)} getrennte Dokumente.")

# --- 2. VEKTOR DB ---
print("💾 Indexiere Dokumente...")
vectorstore = FAISS.from_documents(docs, embedder)

# WICHTIG: Wir holen nur Top 2 Treffer.
# Standard RAG muss sich entscheiden: Twitter-Dokument oder SpaceX-Dokument?
# Es kann wahrscheinlich nicht beide holen, weil die 'Noise'-Dokumente (Jeff Bezos etc.) 
# auch das Wort 'founded' enthalten und stark ablenken.
RETRIEVER_K = 2
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

# --- 3. DIE FRAGE ---
# Die Frage verbindet beide Welten, die im Text getrennt sind.
question = "Welche Firma wurde von dem Besitzer von Twitter gegründet?"
print(f"\n❓ Frage: {question}")

# --- 4. RETRIEVAL CHECK ---
print(f"\n🔍 Standard RAG holt die Top {RETRIEVER_K} Dokumente:")
print("=" * 60)
retrieved = retriever.invoke(question)

found_twitter = False
found_spacex = False

for i, doc in enumerate(retrieved):
    print(f"--- TREFFER {i+1} ---")
    print(doc.page_content)
    if "Twitter" in doc.page_content: found_twitter = True
    if "SpaceX" in doc.page_content: found_spacex = True

print("=" * 60)

# --- 5. ANALYSE ---
if found_twitter and not found_spacex:
    print("❌ BEWEIS ERBRAUCHT: Standard RAG hat 'SpaceX' NICHT gefunden.")
    print("   Grund: Die Vektorsuche hat sich auf 'Twitter' konzentriert und SpaceX ignoriert,")
    print("   weil die Verbindung 'Elon Musk' als Brücke fehlte.")
elif found_twitter and found_spacex:
    print("⚠️  Verdammt, das Modell ist zu gut (oder zu wenig Noise).")
    print("   Versuche, K auf 1 zu setzen oder mehr Noise-Dokumente hinzuzufügen.")

# --- 6. ANTWORT ---
template = """Beantworte die Frage NUR basierend auf dem Kontext.
Kontext: {context}
Frage: {question}
Antwort:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

print("\n🤖 KI Antwort:")
context_str = "\n".join([d.page_content for d in retrieved])
print(chain.invoke({"context": context_str, "question": question}).content)