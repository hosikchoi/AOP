# 3단계: 유사도 기반 추천 후보 간 연결 가능성을 Neo4j로 탐색
-MIE → KE → AO로 연결 가능한 chain을 구성
-관계 스코어를 활용하여 DP 기반으로 최적 경로를 구성

AOP-Wiki(또는 Neo4j)에 존재하는 연결관계(LEADS_TO)를 활용하여 후보 간 연결 여부 확인

연결가능한 path들에 대해 유사도 기반 점수를 매기고

Dynamic Programming (DP) 기반으로 최적 경로를 선택

입력: 추출된 후보 이벤트들 + 유사 AOP 이벤트 ID + 유사도 점수
      (ex: "TGF-β1 증가" → KE:123, score: 0.94)

중간: Neo4j에 등록된 관계 확인
      (ex: KE:101 → KE:123 → AO:999)

출력: 최적화된 path (MIE → KE₁ → KE₂ → ... → AO)
      + Neo4j Cypher 관계 추천

# 3-1. Neo4j에서 관계 확인 (Cypher 쿼리)
// 두 후보 이벤트 간 연결 확인
MATCH (a:Event {id:"KE:101"})-[:LEADS_TO]->(b:Event {id:"KE:123"})
RETURN a.name, b.name

# 3-2. 후보 경로 그래프 구성
예시 후보들
events = [
  {"id": "MIE:001", "text": "PPARα activation", "score": 0.92, "type": "MIE"},
  {"id": "KE:123", "text": "TGF-β1 increase", "score": 0.94, "type": "KE"},
  {"id": "KE:456", "text": "Fibronectin expression", "score": 0.88, "type": "KE"},
  {"id": "AO:789", "text": "Pulmonary fibrosis", "score": 0.91, "type": "AO"}
]
관계 목록
edges = {
  ("MIE:001", "KE:123"): 1.0,
  ("KE:123", "KE:456"): 1.0,
  ("KE:456", "AO:789"): 1.0
}
# 3-3. DP 기반 최적 AOP 경로 선택 (Python 예시)
def find_best_path(events, edges):
    events_by_type = {
        "MIE": [e for e in events if e["type"] == "MIE"],
        "KE":  [e for e in events if e["type"] == "KE"],
        "AO":  [e for e in events if e["type"] == "AO"],
    }

    best_score = -1
    best_path = None

    for mie in events_by_type["MIE"]:
        for ke1 in events_by_type["KE"]:
            for ke2 in events_by_type["KE"]:
                if ke1["id"] == ke2["id"]:
                    continue
                for ao in events_by_type["AO"]:
                    path = [mie, ke1, ke2, ao]
                    ids = [e["id"] for e in path]
                    if all((ids[i], ids[i+1]) in edges for i in range(len(ids)-1)):
                        score = sum(e["score"] for e in path)
                        if score > best_score:
                            best_score = score
                            best_path = path
    return best_path

# 3-4. 최종 결과 예시 (최적 AOP 경로)
1. PPARα activation (MIE:001)
2. → TGF-β1 increase (KE:123)
3. → Fibronectin expression (KE:456)
4. → Pulmonary fibrosis (AO:789)
[총 점수: 3.65]

# 3-5. Neo4j 저장용 Cypher 생성
MERGE (e1:Event {id:"MIE:001", name:"PPARα activation", type:"MIE"})
MERGE (e2:Event {id:"KE:123", name:"TGF-β1 increase", type:"KE"})
MERGE (e3:Event {id:"KE:456", name:"Fibronectin expression", type:"KE"})
MERGE (e4:Event {id:"AO:789", name:"Pulmonary fibrosis", type:"AO"})

MERGE (e1)-[:LEADS_TO {via:"AOP-Wiki"}]->(e2)
MERGE (e2)-[:LEADS_TO {via:"AOP-Wiki"}]->(e3)
MERGE (e3)-[:LEADS_TO {via:"AOP-Wiki"}]->(e4)


















