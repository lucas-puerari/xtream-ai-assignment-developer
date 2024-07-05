from domain.diamond import Diamond
from infrastructure.mlflow_client import MLflowClient
from infrastructure.asset_client import AssetClient


class DiamondService:
    def __init__(self, mlflow_client: MLflowClient, asset_client: AssetClient):
        self.mlflow_client = mlflow_client
        self.asset_client = asset_client

    def estimate_price(self, carat: float, cut: str, color: str, clarity: str, x: float):
        diamond = Diamond(
            carat=carat,
            cut=cut,
            color=color,
            clarity=clarity,
            x=x
        )

        price_estimate = self.mlflow_client.predict_price(diamond)

        return price_estimate

    def find_similar_diamonds(
            self,
            carat: float,
            cut: str,
            color: str,
            clarity: str,
            x: float,
            n: int = 3
        ):
        diamond = Diamond(
            carat=carat,
            cut=cut,
            color=color,
            clarity=clarity,
            x=x
        )

        similar_diamonds = self.asset_client.find_similar_diamonds(diamond, n)

        return similar_diamonds
