import io
import os
import tempfile
from pathlib import Path

import docx2txt
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def parse_resume(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        full_text = "\n".join(
            page.extract_text() for page in reader.pages
        )

    elif ext == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            full_text = docx2txt.process(tmp_path)
        finally:
            os.unlink(tmp_path)

    if not full_text.strip():
        raise ValueError("텍스트 추출 실패 (스캔본일 수 있음)")

    return full_text