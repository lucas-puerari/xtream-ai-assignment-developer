from fastapi import APIRouter, Query, HTTPException, Depends

from application.diamond_service import DiamondService
from infrastructure.mlflow_client import MLflowClient
from infrastructure.asset_client import AssetClient
from domain.diamond import Diamond, CutEnum, ColorEnum, ClarityEnum


router = APIRouter()


mlflow_client = MLflowClient(model_uri="models:/sklearn-XGBRegressor/latest")
asset_client = AssetClient(asset_path="./assets/diamonds.csv")
diamond_service = DiamondService(mlflow_client, asset_client)


def get_diamond_params(
    carat: float = Query(..., gt=0),
    cut: CutEnum = Query(...),
    color: ColorEnum = Query(...),
    clarity: ClarityEnum = Query(...),
    x: float = Query(..., gt=0)
):
    return Diamond(carat=carat, cut=cut, color=color, clarity=clarity, x=x)


@router.get("/diamond/estimate-price/")
async def estimate_price(diamond: Diamond = Depends(get_diamond_params)):
    try:
        price_estimate = diamond_service.estimate_price(
            diamond.carat, diamond.cut, diamond.color, diamond.clarity, diamond.x
        )
        return {"price_estimate": price_estimate}
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


@router.get("/diamond/find-similar/")
async def find_similar_diamonds(
    diamond: Diamond = Depends(get_diamond_params),
    n: int = Query(3, gt=0)
):
    try:
        similar_diamonds = diamond_service.find_similar_diamonds(
            diamond.carat, diamond.cut, diamond.color, diamond.clarity, diamond.x, n
        )
        return {"similar_diamonds": similar_diamonds}
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
