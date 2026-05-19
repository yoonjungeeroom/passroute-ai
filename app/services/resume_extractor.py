import json
import re
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "resume-extract-prompt.md"
)


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def extract_structured_info(raw_text: str) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        template=_load_prompt(),
        input_variables=["text"],
    )

    chain = prompt | llm
    response = await chain.ainvoke({"text": raw_text})

    cleaned = _clean_json(response.content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n응답: {cleaned[:300]}")
    