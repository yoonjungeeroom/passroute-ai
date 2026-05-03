# 세션 요약 프롬프트

## System Prompt

```
당신은 채용 면접 평가 전문가입니다.
면접 세션 전체의 평가 결과를 바탕으로 종합적인 피드백을 생성합니다.
JSON 형식으로만 결과를 반환하며, JSON 외의 텍스트는 포함하지 마세요.
```

---

## User Prompt

```
아래는 면접 세션 전체의 평가 데이터입니다. 종합 피드백을 생성해주세요.

[직무 정보]
- 직무: {job_title}
- 회사: {company_name}

[질문별 요약]
{per_question_summaries}
# 각 질문의 question_index, question_type, question, percentage, summary.strengths, summary.improvements 포함

[항목별 평균 점수]
{item_averages}
# 각 항목의 avg, evaluated_count 포함

[세션 점수]
{session_score}
# raw, percentage, consistency_score 포함

[최고 답변]
{best_q}
# question, percentage, summary.strengths 포함

[최저 답변]
{worst_q}
# question, percentage, summary.improvements 포함

---

## 출력 형식 (JSON)

{
  "overall": "세션 전체에 대한 종합 평가 2~3문장",
  "strengths": "세션 전반에서 반복적으로 잘한 점 1~2문장",
  "improvements": "세션 전반에서 반복적으로 부족한 점 + 개선 방향 1~2문장",
  "question_highlights": [
    {
      "question_index": 최고 답변 인덱스 (정수),
      "type": "best",
      "comment": "최고 답변에 대한 구체적인 코멘트 1문장"
    },
    {
      "question_index": 최저 답변 인덱스 (정수),
      "type": "worst",
      "comment": "최저 답변에 대한 구체적인 코멘트 1문장"
    }
  ]
}
```

---

## 변수 설명

| 변수                         | 설명                                           |
|----------------------------|----------------------------------------------|
| `{job_title}`              | 지원 직무                                        |
| `{company_name}`           | 지원 회사                                        |
| `{per_question_summaries}` | 질문별 strengths/improvements 요약 목록. 아래 구조로 전달. |

```json
[
  {
    "question_index": 1,
    "question_type": "technical",
    "question": "RESTful API 설계 원칙에 대해 설명해주세요.",
    "percentage": 75,
    "summary": {
      "strengths": "기술 개념 설명이 정확하고 구체적입니다.",
      "improvements": "개념 설명에 더해 설계 시 고려할 트레이드오프를 보완하면 좋습니다."
    }
  },
  {
    "question_index": 2,
    "question_type": "personality",
    "question": "갈등 상황을 어떻게 해결했나요?",
    "percentage": 82,
    "summary": {
      "strengths": "경험 기반의 진정성 있는 답변이었습니다.",
      "improvements": "결과 부분을 더 구체적으로 언급하면 좋습니다."
    }
  }
]
```
| `{item_averages}` | Spring에서 계산한 항목별 평균 |
| `{session_score}` | Spring에서 계산한 세션 점수 |
| `{best_q}` | 가장 높은 점수를 받은 질문 정보. 아래 구조로 전달. |
| `{worst_q}` | 가장 낮은 점수를 받은 질문 정보. 아래 구조로 전달. |

```json
{
  "question_index": 2,
  "question": "갈등 상황을 어떻게 해결했나요?",
  "percentage": 82,
  "summary": {
    "strengths": "경험 기반의 진정성 있는 답변이었습니다."
  }
}
```

> `best_q`는 `summary.strengths`, `worst_q`는 `summary.improvements`만 포함.

---

## 참고

- `best_q`, `worst_q`는 Spring에서 계산 후 FastAPI로 전달
- `overall`은 세션 전체 흐름 기반으로 작성 (단순 평균 수치 나열 금지)
- 개선점은 구체적인 행동 방향 포함 권장
