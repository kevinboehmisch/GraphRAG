import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # models/gemini-embedding-001 als alternative
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
# Create retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# Search for similar items
result = retriever.search(query_text="Toys coming alive", top_k=5)

# Parse results
for item in result.items:
    print(item.content, item.metadata["score"])
    
# Close the database connection
driver.close()
