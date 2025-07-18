# 1단계: 논문 텍스트 기반 MIE/KE/AO 이벤트 추출기

# (1) LLM 기반 추출 프롬프트 예시

You are an AOP expert. Given the following scientific paragraph, extract biological events and label them as one of:

- Molecular Initiating Event (MIE)
- Key Event (KE)
- Adverse Outcome (AO)

Return a JSON list of extracted events with their types and brief rationale.

Text:
"Pycnogenol attenuated PHMG-induced TGF-β1 increase, reducing fibronectin and collagen deposition, ultimately alleviating pulmonary fibrosis in mice."

Output:
[
  {
    "event": "TGF-β1 increase",
    "type": "KE",
    "reason": "Intermediate signaling molecule altered by stressor"
  },
  {
    "event": "fibronectin deposition",
    "type": "KE",
    "reason": "ECM remodeling involved in tissue-level change"
  },
  {
    "event": "pulmonary fibrosis",
    "type": "AO",
    "reason": "Terminal adverse pathology in the lung"
  }
]
# (2) Python 코드 예시 (OpenAI API 기반)
import openai
import json

openai.api_key = "your-api-key"

def extract_aop_events(text):
    prompt = f"""
You are an AOP expert. Given the following scientific paragraph, extract biological events and label them as MIE, KE, or AO.
Return a JSON list of extracted events with their types and brief rationale.

Text:
\"\"\"{text}\"\"\"
"""
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    try:
        output = response.choices[0].message['content']
        return json.loads(output)
    except Exception as e:
        print("Error parsing response:", e)
        return None

# (3) 예시 실행

text = """
Pycnogenol treatment reduced PHMG-induced TGF-β1 expression, inhibited fibronectin accumulation,
and mitigated the progression of lung fibrosis in mouse models.
"""

events = extract_aop_events(text)
for e in events:
    print(f"{e['type']}: {e['event']} – {e['reason']}")

# 출력 예시

[
  {
    "event": "TGF-β1 expression",
    "type": "KE",
    "reason": "Cytokine mediating fibrogenic signaling"
  },
  {
    "event": "fibronectin accumulation",
    "type": "KE",
    "reason": "ECM component buildup linked to fibrosis"
  },
  {
    "event": "lung fibrosis",
    "type": "AO",
    "reason": "Adverse pathological outcome in lung"
  }
]


















