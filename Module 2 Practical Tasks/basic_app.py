from fastapi import FastAPI

app = FastAPI(
    title="My First API",
    description="A simple API built with FastAPI",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to my API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}