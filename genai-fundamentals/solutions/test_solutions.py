def test_vector_retriever(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "vector_retriever",
    )
    
    assert output > ""

def test_vector_rag(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "vector_rag",
    )
    
    assert output > ""

def test_vector_cypher_rag(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "vector_cypher_rag",
    )
    
    assert output > ""

def test_text2cypher_rag(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "text2cypher_rag",
    )
    
    assert output > ""

def test_text2cypher_rag_examples(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "text2cypher_rag_examples",
    )
    
    assert output > ""

def test_text2cypher_rag_schema(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "text2cypher_rag_schema",
    )
    
    assert output > ""