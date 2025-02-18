from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import ann

app = FastAPI(title="Artificial Neural Network")

# CORS settings
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.openapi_schema = None
app.include_router(ann.router)
