import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
# tag::import-llm[]
from neo4j_graphrag.llm import OpenAILLM
# end::import-llm[]
# tag::import-graphrag[]
from neo4j_graphrag.generation import GraphRAG
# end::import-graphrag[]

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# tag::llm[]
# Create the LLM
llm = OpenAILLM(model_name="gpt-4o")
# end::llm[]

# tag::llm-temp[]
# Modify the LLM configuration if needed
llm = OpenAILLM(
    model_name="gpt-3.5-turbo", 
    model_params={"temperature": 1}
)
# end::llm-temp[]

# tag::graphrag[]
# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)
# end::graphrag[]

# tag::search[]
# Search
query_text = "Find me movies about toys coming alive"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5}
)

print(response.answer)
# end::search[]

# tag::search_return_context[]
# Search
query_text = "Find me movies about toys coming alive"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)
# end::search_return_context[]

# Close the database connection
driver.close()