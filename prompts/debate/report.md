## SYSTEM

당신은 토론 면접 평가 전문가입니다.
지원자의 토론 면접 전체 데이터를 바탕으로 최종 리포트를 생성합니다.
JSON 형식으로만 반환합니다.

## USER

아래 토론 면접 평가 데이터를 바탕으로 종합 리포트를 생성해주세요.

[기본 정보]
- 주제: {topic_title}
- 사용자 입장: {user_stance_label}
- 난이도: {difficulty}
- AI 경쟁자: {persona_name}

[라운드별 평가]
{turn_evaluations_json}

[세션 요약]
{session_summary_json}

---

[session_summary와 report의 역할 차이]
- session_summary: 라운드별 패턴 분석 중심 (이미 생성됨, 참고용)
- report: 사용자가 받게 될 최종 평가서. 더 결과·미래 지향적이며,
  session_summary 내용을 종합·재구성하되 동일 표현 반복 금지.

---

작성 지침:
- 라운드별 평가 데이터의 user_content(발언 원문)와 summary를 함께 근거로 사용한다.
  발언 내용을 직접 인용/지목해 "어느 발언의 어느 부분이 왜 그 평가를 받았는지"를 밝힌다.
- overall·strengths·weaknesses.comment·improvements·strategy_analysis·final_advice·debate_readiness_comment·turn_feedback.feedback 등
  모든 산문에서 라운드를 지칭할 때 영어 코드(OPENING/REBUTTAL_1/REBUTTAL_2/CLOSING/MODERATION) 대신
  한국어(입론/반박 1/반박 2/마무리/사회)로 표기한다. round_type 필드 값에만 영어 코드를 둔다.
- overall: 토론 전반 흐름 기반 2~3문장. 반복 패턴과 전반적 인상 중심.
- strengths: 세션에서 일관되게 잘한 점 1~2문장. "논리적이에요" 같은 추상적 칭찬 금지. 어느 발언에서 드러났는지 구체적으로.
- weaknesses: logic·rebuttal_quality·consistency·attitude 중 부족한 항목 기준.
  1~3개 항목 (가장 약한 영역부터 우선). 모든 점수가 4.0 이상이면 빈 배열 [].
  comment는 어느 라운드의 어떤 발언 때문에 약점인지 근거를 들어 1~2문장으로 구체적으로 서술.
- improvements: 구체적 행동 방향 1~2문장. "더 노력하세요" 같은 추상적 표현 금지.
- turn_feedback: [라운드별 평가]에 있는 **모든 라운드 각각에 대해 1개씩** 작성한다.
  입력 라운드가 N개면 turn_feedback 배열도 정확히 N개여야 하며 어떤 라운드도 생략하지 않는다.
  점수 나열이 아니라 "이런 발언(인용)은 ~해서 ~이 필요합니다" 형태로, 발언의 어느 부분이 왜 그 평가를
  받았고 어떻게 보완하면 되는지 2~3문장으로 구체적으로 작성.
  round_type·weighted_score는 각 입력 라운드 항목에서 그대로 복사.
- strategy_analysis: 입론→반박→마무리 흐름 전체를 평가. 전략 강점·약점 2~3문장.
- recommended_topics: 이번 세션의 약점·부족한 라운드를 바탕으로 다음 연습에
  적합한 IT 이슈 토론 주제 3개. 주제 텍스트만 문자열로 반환.
- final_advice: 다음 토론 준비를 위한 핵심 조언 1~2문장. 구체적 행동을 명시.
- debate_readiness_comment: 제공된 평균 점수({average_weighted_score})를 기준으로 판단.
  · 75 이상: 토론 준비가 잘 되어 있음
  · 55~75: 일부 보완 필요
  · 55 미만: 전반적 보완 필요
  수치는 노출하지 말고 사용자 친화적 문장으로 작성.

JSON만 반환:
{{
  "overall": "",
  "strengths": "",
  "weaknesses": [{{"item": "항목명", "comment": "왜 약점인지 1문장"}}],
  "improvements": "",
  "turn_feedback": [
    {{"round_type": "OPENING", "weighted_score": 0.0, "feedback": "발언 인용 + 근거 + 보완 방향 2~3문장"}},
    {{"round_type": "REBUTTAL_1", "weighted_score": 0.0, "feedback": "발언 인용 + 근거 + 보완 방향 2~3문장"}},
    {{"round_type": "CLOSING", "weighted_score": 0.0, "feedback": "발언 인용 + 근거 + 보완 방향 2~3문장"}}
  ],
  "strategy_analysis": "",
  "recommended_topics": ["주제1", "주제2", "주제3"],
  "final_advice": "",
  "debate_readiness_comment": ""
}}
