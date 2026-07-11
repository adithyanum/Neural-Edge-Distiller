from fastapi import FastAPI

app = FastAPI(title="Neural Edge Distiller — Control Plane")

@app.get("/health")
def health():
    return {"status": "ok"}