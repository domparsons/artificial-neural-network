from fastapi import APIRouter
from app.ann import main

router = APIRouter(
    prefix="/artificial-neural-network", tags=["artificial-neural-network"]
)


@router.get("/")
def run_ann():
    correlation, precision, mae = main()
    return {"correlation": correlation, "precision": precision, "mae": mae}
