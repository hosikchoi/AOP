# Step 1: Node Sentence Embedding (SBERT 사용)
from sentence_transformers import SentenceTransformer
import pandas as pd

# 예시 문장 (실제로는 paper2event.py에서 추출한 MIE/KE/AO 문장 사용)
events = {
    "n1": "Pycnogenol inhibits TGF-β1 expression",
    "n2": "Fibronectin production is reduced",
    "n3": "Collagen deposition is decreased",
    "n4": "Pulmonary fibrosis is alleviated"
}

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = {k: model.encode(v) for k, v in events.items()}

# Step 2: Edge weight = Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
import itertools

# Create edge weight matrix
edge_list = []
for src, tgt in itertools.permutations(events.keys(), 2):
    sim = cosine_similarity([embeddings[src]], [embeddings[tgt]])[0][0]
    edge_list.append((src, tgt, sim))

###########
# LLM을 이용하여 두 문장 간 인과 관계 및 plausibility를 판단하는 방식
# Convert to DataFrame
edges_df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
edges_df = edges_df[edges_df["weight"] > 0.5]  # 임계값 필터링
edges_df.to_csv("edges_embedded.csv", index=False)

import openai
openai.api_key = "sk-..."

def score_relation(source_sent, target_sent):
    prompt = f"""
Given the following two event sentences from toxicology literature:

1. {source_sent}
2. {target_sent}

Is sentence 1 causally linked to sentence 2? If yes, how plausible is this relationship on a scale from 0 to 1?

Answer in this format only: "Yes, plausibility: 0.78" or "No, plausibility: 0.00"
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["choices"][0]["message"]["content"]
    score = float(answer.split("plausibility:")[1].strip())
    return score

relations = []
for src, tgt in itertools.permutations(events.keys(), 2):
    src_sent, tgt_sent = events[src], events[tgt]
    weight = score_relation(src_sent, tgt_sent)
    if weight > 0.5:
        relations.append((src, tgt, weight))

edges_llm = pd.DataFrame(relations, columns=["source", "target", "weight"])
edges_llm.to_csv("edges_llm.csv", index=False)







