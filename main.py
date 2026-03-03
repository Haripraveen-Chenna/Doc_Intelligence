from fastapi import FastAPI
from doc_tracker_route import router as doc_router

app = FastAPI(
    title="Document Project API",
    version="1.0.0"
)

# Include your router that has:
# - /generate-tasks
# - /generate-projects
app.include_router(
    doc_router,
    prefix="/api",   # optional
    tags=["Projects"]
)


@app.get("/")
def health_check():
    return {"status": "API is running"}