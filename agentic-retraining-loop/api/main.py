from fastapi import FastAPI
from api.routes import router
from db.session import init_db

app = FastAPI(title="Agentic Retraining Loop", version="0.1.0")
app.include_router(router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}
