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

# 4단
### Neo4j에서 노드/엣지 가져오기
import networkx as nx
from neo4j import GraphDatabase

def build_graph_from_neo4j(driver):
    G = nx.DiGraph()
    with driver.session() as session:
        # 노드 불러오기
        nodes = session.run("MATCH (e:Event) RETURN e.id AS id, e.text AS text")
        for row in nodes:
            G.add_node(row["id"], text=row["text"])

        # 관계 불러오기 (모든 관계 유형 포함)
        edges = session.run("""
            MATCH (a:Event)-[r]->(b:Event)
            RETURN a.id AS source, b.id AS target, type(r) AS rel_type, r.weight AS weight
        """)
        for row in edges:
            G.add_edge(row["source"], row["target"], weight=row["weight"], type=row["rel_type"])
    return G

# DP 최적화 수행 (최적 path: MIE → AO)

def find_mie_and_ao_nodes(G):
    mie = [n for n in G.nodes if "mie" in G.nodes[n].get("text", "").lower()]
    ao = [n for n in G.nodes if "adverse" in G.nodes[n].get("text", "").lower()]
    return mie, ao

def dp_optimal_path(G, source, target):
    topo_order = list(nx.topological_sort(G))
    f = {v: float("-inf") for v in G.nodes}
    prev = {v: None for v in G.nodes}
    f[source] = 0

    for v in topo_order:
        for u in G.predecessors(v):
            w = G[u][v]["weight"]
            if f[u] + w > f[v]:
                f[v] = f[u] + w
                prev[v] = u

    path = []
    current = target
    while current:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path, f[target]

# 통합 실행 예시
G = build_graph_from_neo4j(driver)
mie_nodes, ao_nodes = find_mie_and_ao_nodes(G)

# 모든 MIE → AO 조합에 대해 DP 경로 계산
for src in mie_nodes:
    for tgt in ao_nodes:
        path, score = dp_optimal_path(G, src, tgt)
        if len(path) > 1:
            print("AOP chain:", " -> ".join(path))
            print("Score:", round(score, 3))

AOP chain: e1 -> e2 -> e3
Score: 2.68

Streamlit으로 경로 추천 인터페이스


















