from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import os

# Create the FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import API routes
from app.api.routes import router as api_router

# Add routes to the app
app.include_router(api_router, prefix="/api")

# Ensure data directories exist
@app.on_event("startup")
async def startup_event():
    os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": settings.API_VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
