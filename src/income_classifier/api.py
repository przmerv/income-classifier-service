from fastapi import FastAPI

app = FastAPI(
    title="income-classifier-service",
    description="A service that classifies income based on user data",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Health check endpoint to verify that the service is running.
    Returns a simple JSON response indicating the service status.
    """
    return {"status": "ok"}
