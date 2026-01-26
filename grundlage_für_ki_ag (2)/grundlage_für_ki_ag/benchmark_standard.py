import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- SETUP ---
FILE_PATH = "long_data.txt" # DEINE DATEI HIER REINLEGEN
MODEL_NAME = "gemma3:27b"

print(f"🚀 Starte STANDARD RAG (Token Split 1200/100) mit {MODEL_NAME}...")

if not os.path.exists(FILE_PATH):
    print(f"❌ Datei '{FILE_PATH}' nicht gefunden! Bitte lege eine lange .txt Datei ab.")
    exit()

llm = ChatOllama(model=MODEL_NAME, temperature=0)
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- 1. LADEN & SPLITTEN (NACH PAPIER-STANDARD) ---
print("📥 Lade Textdatei...")
loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()

print("🔪 Token-Splitting (1200 Tokens, 100 Overlap)...")
# Das ist der entscheidende Teil aus deinem Beispiel
text_splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

print(f"   -> {len(splits)} Chunks erstellt.")
print(f"   -> Beispiel Chunk-Länge: {len(splits[0].page_content)} Zeichen")

# --- 2. INDEXIERUNG ---
print("💾 Erstelle Vektor-Index (FAISS)...")
start_time = time.time()
vectorstore = FAISS.from_documents(splits, embedder)
end_time = time.time()
print(f"   ✅ Indexierung fertig in {end_time - start_time:.2f} Sekunden.")

# Wir holen Top 3 (das sind dann 3600 Tokens Kontext - sehr viel!)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. DIE FRAGE ---
# WICHTIG: Stelle hier eine Frage, deren Antwort über den ganzen Text verstreut ist!
question = "Was sind die wichtigsten Herausforderungen und die vorgeschlagenen Lösungen im Text?"
# ODER spezifischer, wenn du den Inhalt kennst:
# question = "Welche Beziehung besteht zwischen Konzept A (Seite 1) und Konzept B (Seite 50)?"

print(f"\n❓ Frage: {question}")

# --- 4. RETRIEVAL & ANTWORT ---
print("🔍 Suche relevante Chunks...")
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

template = """Du bist ein Experte. Beantworte die Frage basierend auf dem Kontext.
Kontext:
{context}

Frage: {question}
Antwort:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

print("\n🤖 Standard RAG Antwort:")
print("="*60)
response = chain.invoke({"context": context_text, "question": question})
print(response.content)
print("="*60)