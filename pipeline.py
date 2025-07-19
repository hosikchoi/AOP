from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# Step 1: 문헌에서 Event 후보 추출 (paper2event.py 출력 등 사용)
events = {
    "e1": "Pycnogenol inhibits TGF-β1",
    "e2": "Collagen production is reduced",
    "e3": "Fibrosis is suppressed"
}

# Step 2: 문장 embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = {k: model.encode(v) for k, v in events.items()}

# Step 3: Neo4j에서 k-hop context RAG
def get_context_from_neo4j(event_text, top_k=3):
    with driver.session() as session:
        query = """
        MATCH (n:Event)
        WHERE toLower(n.text) CONTAINS toLower($text)
        CALL {
            WITH n
            MATCH (n)-[:CAUSES|ACTIVATES|INHIBITS*1..2]->(m)
            RETURN m.text AS related
            LIMIT $k
        }
        RETURN collect(related) AS context
        """
        result = session.run(query, text=event_text, k=top_k)
        return result.single()["context"]

# 예시 context
rag_context = get_context_from_neo4j(events["e1"])
