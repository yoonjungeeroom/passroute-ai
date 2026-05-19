from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.resume_parser import parse_resume
from app.services.resume_vector_store import store_resume, search_candidates

router = APIRouter(prefix="/resume", tags=["resume"])


@router.post("/process")
async def process_resume(user_id: str, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        raw_text = parse_resume(file_bytes, file.filename)
        store_resume(user_id, raw_text)

        return {"status": "success", "user_id": user_id}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")


@router.get("/search")
def search(query: str, top_k: int = 5):
    results = search_candidates(query, top_k)
    return {"query": query, "results": results}