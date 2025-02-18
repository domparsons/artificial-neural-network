from fastapi import APIRouter
from app.api.endpoints import ann

api_router = APIRouter()
api_router.include_router(ann.router, prefix="/ann", tags=["ann"])
