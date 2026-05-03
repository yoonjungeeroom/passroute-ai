# 점수 계산 정책 (Score Policy)

## 1. 가중치 테이블

| 항목              | technical | personality |
|-----------------|-----------|-------------|
| `relevance`     | 0.15      | 0.15        |
| `logic`         | 0.15      | 0.20        |
| `specificity`   | 0.15      | 0.15        |
| `conciseness`   | 0.10      | 0.15        |
| `clarity`       | 0.10      | 0.15        |
| `accuracy`      | 0.15      | null        |
| `depth`         | 0.15      | null        |
| `job_relevance` | 0.05      | 0.05        |
| `authenticity`  | null      | 0.10        |
| `growth`        | null      | 0.05        |
| **합계**          | **1.00**  | **1.00**    |

---

## 2. 질문 단위 점수 계산

### 2-1. voice penalty

음성 분석 결과에 따라 `conciseness` 점수에 감점 적용.

#### filler_word_count 감점

`filler_word_count` 감점은 구간형으로 적용하며, 둘 중 하나만 적용한다.

| 조건                        | 감점   |
|---------------------------|------|
| `filler_word_count >= 10` | -1.0 |
| `filler_word_count >= 5`  | -0.5 |
| `filler_word_count < 5`   | 0    |

> `filler_word_count >= 10`이면 -1.0만 적용한다.  
> -1.0과 -0.5를 중복 적용하지 않는다.

#### speech_rate_diff 감점

`speech_rate_diff` 감점은 filler 감점과 별도로 독립 적용한다.

| 조건 | 감점             |
| ---- | ---------------- | ------ | ---- |
| `    | speech_rate_diff | >= 20` | -0.5 |
| `    | speech_rate_diff | < 20`  | 0    |

#### silence_count 처리

`silence_count`는 v1에서는 점수 감점에 직접 반영하지 않고, 음성 피드백 생성 시 참고 정보로만 사용한다.

추후 기준이 확정되면 penalty 항목으로 확장할 수 있다.

#### 최종 voice penalty 계산

```text
voice_penalty = filler_penalty + speech_rate_penalty
```

예시:

| filler_word_count | speech_rate_diff | filler_penalty | speech_rate_penalty | voice_penalty |
|------------------:|-----------------:|---------------:|--------------------:|--------------:|
|                 3 |               10 |              0 |                   0 |             0 |
|                 6 |               10 |           -0.5 |                   0 |          -0.5 |
|                 6 |               25 |           -0.5 |                -0.5 |          -1.0 |
|                11 |               10 |           -1.0 |                   0 |          -1.0 |
|                11 |               25 |           -1.0 |                -0.5 |          -1.5 |

---

### 2-2. conciseness_final

```
conciseness_final = max(1.0, llm_score["conciseness"] + voice_penalty)
```

- `conciseness_final`은 최소 1.0을 보장한다.
- 전체 점수 계산 시 `conciseness` 항목은 LLM 원점수가 아니라 `conciseness_final`을 사용한다.

---

### 2-3. 총점 계산

질문 단위 점수는 유효한 평가 항목의 weight 합계를 기준으로 재정규화한다.

```
valid_weights_sum = Σ (평가된 항목의 weight)
raw = Σ (effective_score × weight) / valid_weights_sum
단, conciseness는 conciseness_final 사용

percentage = (raw / 5.0) × 100
```

#### effective_score 기준

```text
if item == "conciseness":
    effective_score = conciseness_final
else:
    effective_score = llm_scores[item].score
```

#### null 항목 처리

- `null` 항목은 점수 계산에서 제외한다.
- `null` 항목의 weight도 `valid_weights_sum`에서 제외한다.
- 현재 `technical` / `personality`의 유효 weight 합계는 1.00이지만, 추후 질문 유형 확장 또는 일부 항목 평가 실패 상황을 고려해 재정규화 방식을 기본 정책으로 사용한다.

---

## 3. 세션 단위 점수 계산

### 3-1. item_averages

각 평가 항목별 평균을 계산한다.

- null이 아닌 질문만 평균 계산에 포함한다.
- `conciseness`는 반드시 `conciseness_final` 기준으로 평균 계산한다.
- 평가된 질문 수는 `evaluated_count`로 함께 저장한다.

```
item_avg = Σ(item_score) / evaluated_count
```

---

### 3-2. session_score

#### 단일 타입 세션 (모든 질문이 동일한 question_type)

유효한 항목의 weight 합계를 기준으로 재정규화한다.

```
valid_weights_sum = Σ(평가된 항목의 weight)
session_raw = Σ (item_avg × weight) / valid_weights_sum
percentage = (session_raw / 5.0) × 100
```

#### 혼합 타입 세션 (technical / personality 질문이 혼재)

각 질문의 `percentage`는 이미 해당 타입의 weight가 적용된 값이므로, 질문별 percentage의 단순 평균을 세션 점수로 사용한다.

```
session_percentage = Σ(question_percentage) / total_questions
```

> `item_averages`에서 weight를 재적용하면 `conciseness`, `logic` 등 타입별로 weight가 다른 항목에서 왜곡이 발생한다.  
> 혼합 세션에서는 질문 단위 percentage 평균 방식을 기본으로 사용한다.

---

### 3-3. consistency_score

초기 버전에서는 질문별 percentage의 최대/최소 차이를 기반으로 일관성 점수를 계산한다.

```
consistency_score = 1 - ((max_percentage - min_percentage) / 100)
```

- `max_percentage`: 세션 내 질문별 점수 중 최고 점수
- `min_percentage`: 세션 내 질문별 점수 중 최저 점수

> v1에서는 range 기반으로 계산한다.  
> 추후 질문 수가 충분히 쌓이면 표준편차 기반 일관성 점수로 개선할 수 있다.

---

## 4. 면접 준비도 판단

> `interview_readiness`는 실제 합격/불합격 판단이 아니라,
> 면접 답변 품질에 대한 시스템 내부 참고 지표로만 사용한다.

| 조건                                            | 판정                  |
|-----------------------------------------------|---------------------|
| `percentage >= 75` AND `min(item_avg) >= 3.0` | `READY `            |
| `percentage >= 55`                            | `NEEDS_REVIEW`      |
| 그 외                                           | `NEEDS_IMPROVEMENT` |

```
confidence = round(percentage / 100, 2)
```

---

## 5. key_weakness 산출

```
if (max_avg - min_avg < 0.5):
    key_weakness = "뚜렷한 약점 없음"
else:
    key_weakness = item_averages 하위 2개 항목
```

- 평균 차이가 크지 않은 경우 특정 약점을 과도하게 지적하지 않는다.
- 약점 항목은 리포트 개선 제안 생성 시 활용한다.
- `key_weakness` 값은 한국어 표기명으로 저장한다.

| 필드명             | 한국어 표기명 |
|-----------------|---------|
| `relevance`     | 질문 적합성  |
| `logic`         | 논리성     |
| `specificity`   | 구체성     |
| `conciseness`   | 간결성     |
| `clarity`       | 명확성     |
| `accuracy`      | 기술 정확성  |
| `depth`         | 기술적 깊이  |
| `job_relevance` | 직무 연관성  |
| `authenticity`  | 진정성     |
| `growth`        | 성장 가능성  |
