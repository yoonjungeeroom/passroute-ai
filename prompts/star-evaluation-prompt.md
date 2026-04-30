# STAR 판단 프롬프트

## System Prompt

```
당신은 면접 답변 구조 분석 전문가입니다.
지원자의 답변에 STAR 구조(Situation, Task, Action, Result)가 포함되어 있는지 판단하고,
JSON 형식으로만 결과를 반환합니다. JSON 외의 텍스트는 포함하지 마세요.
```

---

## User Prompt

```
아래 면접 답변에서 STAR 구조 각 요소가 포함되어 있는지 판단해주세요.

[질문]
{question}

[답변]
{answer}

---

## STAR 구조 기준

- Situation: 배경 상황이나 맥락을 설명했는가
- Task: 본인의 역할이나 해결해야 할 과제를 언급했는가
- Action: 실제로 취한 행동이나 방법을 구체적으로 설명했는가
- Result: 결과나 성과를 언급했는가

---

## 출력 형식 (JSON)

### STAR 평가 적용 대상인 경우 (applicable=true)

```json
{
  "star_evaluation": {
    "applicable": true,
    "reason": "경험 기반 답변으로 STAR 평가 대상입니다.",
    "star_breakdown": {
      "situation": { "present": true,  "feedback": "상황 설명이 구체적으로 포함되어 있습니다." },
      "task":      { "present": true,  "feedback": "해결해야 할 과제가 명확히 드러납니다." },
      "action":    { "present": true,  "feedback": "본인의 행동이 구체적으로 제시됩니다." },
      "result":    { "present": false, "feedback": "결과나 성과에 대한 언급이 부족합니다." }
    },
    "star_score": 3
  }
}
```

### STAR 평가 비적용 대상인 경우 (applicable=false)

```json
{
  "star_evaluation": {
    "applicable": false,
    "reason": "기술 개념 설명형 질문이므로 STAR 평가 대상이 아닙니다.",
    "star_breakdown": null,
    "star_score": null
  }
}
```


---

## 변수 설명

| 변수           | 설명        |
|--------------|-----------|
| `{question}` | 면접 질문 원문  |
| `{answer}`   | 지원자 답변 원문 |

---

## 참고

- **개념 설명형 기술 질문** (예: "RESTful API란?", "OOP 원칙을 설명하세요")은 `applicable=false`로 판단하고 `star_breakdown=null` 반환
- 경험/행동 기반 질문 (예: "~했던 경험을 말해주세요", "갈등을 어떻게 해결했나요?")은 `applicable=true`로 판단
- `applicable` 판단이 애매한 경우 질문에 경험/행동 서술이 요구되는지 여부를 기준으로 결정
- 명시적 언급이 없어도 내용상 포함된다고 판단되면 true로 처리
- star_score는 present: true 항목의 합계 (0~4)
