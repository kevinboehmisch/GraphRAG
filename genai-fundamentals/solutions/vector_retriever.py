import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
# tag::import-embedder[]
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
# end::import-embedder[]
# tag::import-retriever[]
from neo4j_graphrag.retrievers import VectorRetriever
# end::import-retriever[]

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# tag::embedder[]
# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
# end::embedder[]

# tag::retriever[]
# Create retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)
# end::retriever[]

# tag::search[]
# Search for similar items
result = retriever.search(query_text="Toys coming alive", top_k=5)
# end::search[]

# tag::print-results[]
# Parse results
for item in result.items:
    print(item.content, item.metadata["score"])
# end::print-results[]

# Close the database connection
driver.close()
