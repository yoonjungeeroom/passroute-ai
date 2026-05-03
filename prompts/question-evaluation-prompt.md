# 질문 단위 LLM 평가 프롬프트

## System Prompt

```
당신은 채용 면접 평가 전문가입니다.
지원자의 면접 답변을 주어진 기준에 따라 항목별로 평가하고, JSON 형식으로 결과를 반환합니다.
반드시 아래 출력 형식을 정확히 따라야 하며, JSON 외의 텍스트는 포함하지 마세요.
```

---

## User Prompt

```
아래 면접 답변을 평가해주세요.

[채용 정보]
- 직무: {job_title}
- 회사: {company_name}
- JD 키워드: {jd_keywords}

[질문 유형]
{question_type}  # technical 또는 personality

[질문]
{question}

[답변]
{answer}

---

## 평가 항목 및 기준

각 항목을 1~5점으로 평가하고, 간단한 피드백을 작성하세요.

### 공통 항목 (모든 질문에 적용)
- relevance: 질문 의도에 맞는 답변인지
- logic: 답변의 구조와 논리적 흐름
- specificity: 구체적인 경험/수치/예시 포함 여부
- conciseness: 핵심만 간결하게 전달했는지
- clarity: 표현이 명확하고 이해하기 쉬운지
- job_relevance: JD 키워드 및 직무와의 연관성

### 기술 질문 전용 (question_type = technical일 때만 평가)
- accuracy: 기술 개념의 정확성
- depth: 기술적 깊이 (트레이드오프, 내부 동작 등)

### 인성 질문 전용 (question_type = personality일 때만 평가)
- authenticity: 답변의 진정성 (경험 기반 여부)
- growth: 성장 가능성 및 회고 역량

### 점수 기준
- 5: 매우 우수. 기대 이상의 깊이와 구체성
- 4: 우수. 질문 의도에 맞고 충분히 전달됨
- 3: 보통. 핵심은 전달되나 부족한 부분 있음
- 2: 미흡. 핵심이 불분명하거나 구조가 약함
- 1: 불충분. 질문 의도와 맞지 않거나 내용이 거의 없음

---

## 출력 형식 (JSON)

- question_type이 technical이면 authenticity, growth는 null로 설정하세요.
- question_type이 personality이면 accuracy, depth는 null로 설정하세요.
- score는 반드시 1~5 사이의 정수로 반환하고, 소수점은 사용하지 마세요.

아래는 question_type=technical인 경우의 예시입니다.

```json
{
  "llm_scores": {
    "relevance":     { "score": 4, "feedback": "질문 의도에 맞게 REST 원칙을 설명했습니다." },
    "logic":         { "score": 3, "feedback": "답변 흐름이 일부 나열식이며 구조화가 부족합니다." },
    "specificity":   { "score": 4, "feedback": "HTTP 메서드와 상태코드 등 구체적인 예시가 포함되어 있습니다." },
    "conciseness":   { "score": 3, "feedback": "핵심 전달은 되나 일부 군더더기 표현이 있습니다." },
    "clarity":       { "score": 4, "feedback": "표현이 비교적 명확하게 전달됩니다." },
    "accuracy":      { "score": 5, "feedback": "stateless, 캐시 가능성 등 핵심 개념이 정확합니다." },
    "depth":         { "score": 3, "feedback": "기본 개념 설명은 있으나 트레이드오프 언급이 부족합니다." },
    "job_relevance": { "score": 4, "feedback": "JD의 API 설계 경험 키워드와 연결됩니다." },
    "authenticity":  null,
    "growth":        null
  },
  "summary": {
    "strengths": "기술 개념의 정확성과 구체적 예시 활용이 뛰어납니다.",
    "improvements": "답변 구조를 다듬고 트레이드오프를 언급하면 깊이가 더해질 것 같습니다."
  }
}
```

> **weight 필드 처리 방침**
> LLM은 score + feedback만 반환. weight는 FastAPI에서 score-policy.md의 가중치 테이블을 참조하여 응답에 병합.
> LLM에게 weight를 직접 출력하게 하지 않음 (고정값이므로 하드코딩 처리가 더 안전).
```

---

## 변수 설명

| 변수 | 설명 |
|------|------|
| `{job_title}` | 지원 직무 (예: 백엔드 개발자) |
| `{company_name}` | 지원 회사 (예: 카카오) |
| `{jd_keywords}` | JD에서 추출한 키워드 목록 (예: ["REST API", "MSA", "Java"]) |
| `{question_type}` | 질문 유형 (`technical` / `personality`) |
| `{question}` | 면접 질문 원문 |
| `{answer}` | 지원자 답변 원문 |
