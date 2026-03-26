from fastapi import FastAPI

app = FastAPI(
    title = "PassRoute AI server",
    version = "1.0.0"
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}