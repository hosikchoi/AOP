from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

#################################################################
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

####################################################
#2단계: Fine-tuned classifier + RAG 기반 입력 구성

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("your-finetuned-model")
model = AutoModelForSequenceClassification.from_pretrained("your-finetuned-model")

relation_labels = ["ACTIVATES", "INHIBITS", "CAUSES", "ASSOCIATED_WITH", "NO_RELATION"]

def predict_relation_type_and_score(event_a, event_b, context):
    prompt = f"""
Event A: {event_a}
Event B: {event_b}
Context from AOP Graph: {" | ".join(context)}

What is the relation type between A and B?
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = relation_labels[torch.argmax(probs)]
    confidence = probs.max().item()
    return pred_label, confidence
    
# 예시 실행
for src, tgt in [("e1", "e2"), ("e2", "e3")]:
    ctx = get_context_from_neo4j(events[src])
    rel_type, score = predict_relation_type_and_score(events[src], events[tgt], ctx)
    print(f"{src} → {tgt} = {rel_type} ({round(score, 3)})")

def upload_to_neo4j(source_id, target_id, rel_type, weight, source_text, target_text):
    with driver.session() as session:
        session.run("""
        MERGE (a:Event {id: $src})
        SET a.text = $src_text
        MERGE (b:Event {id: $tgt})
        SET b.text = $tgt_text
        MERGE (a)-[r:%s]->(b)
        SET r.weight = $weight
        """ % rel_type, {
            "src": source_id,
            "tgt": target_id,
            "src_text": source_text,
            "tgt_text": target_text,
            "weight": float(weight)
        })
# 전체 자동화
for (src, tgt) in [("e1", "e2"), ("e2", "e3")]:
    ctx = get_context_from_neo4j(events[src])
    rel_type, score = predict_relation_type_and_score(events[src], events[tgt], ctx)
    upload_to_neo4j(src, tgt, rel_type, score, events[src], events[tgt])





















