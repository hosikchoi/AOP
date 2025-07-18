#####################################################
# 2-1. AOP-Wiki 이벤트 목록 수집
#####################################################
AOP-Wiki API 또는 [CSV Export (https://aopwiki.org/downloads)] 에서 다음 정보 확보
json 파일
{
  "id": "KE:123",
  "name": "Increase in TGF-beta1",
  "type": "KE",
  "description": "TGF-β1 is a key cytokine that modulates fibrogenesis..."
}
#####################################################
# 2-2. 임베딩 생성 (E5 or BGE 모델 추천)
#####################################################
pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("intfloat/e5-base-v2")

# AOP-Wiki 이벤트 문장들
aop_events = [
    {"id": "KE:123", "text": "Increase in TGF-beta1"},
    {"id": "KE:456", "text": "Collagen deposition in lung"},
    {"id": "AO:789", "text": "Pulmonary fibrosis"}
]

texts = ["passage: " + e["text"] for e in aop_events]
vectors = model.encode(texts, convert_to_numpy=True)

# FAISS 인덱스 구축
dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors)

#####################################################
#2-3. 추출 이벤트에 대해 유사 AOP 이벤트 검색
#####################################################
query = "TGF-β1 expression"
query_vec = model.encode(["query: " + query])
scores, indices = index.search(query_vec, k=3)

print("Top matches:")
for i in range(len(indices[0])):
    idx = indices[0][i]
    score = scores[0][i]
    print(f"{aop_events[idx]['id']}: {aop_events[idx]['text']} (score={score:.4f})")

### 출력 예시
Top matches:
KE:123: Increase in TGF-beta1 (score=0.94)
KE:456: Collagen deposition in lung (score=0.74)
AO:789: Pulmonary fibrosis (score=0.69)

# 연결 포맷 (Neo4j 매핑 준비용)
[
  {
    "event_candidate": "TGF-β1 expression",
    "matched_event": "Increase in TGF-beta1",
    "aop_id": "KE:123",
    "similarity": 0.94
  },
  ...
]



























