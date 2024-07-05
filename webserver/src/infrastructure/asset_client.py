import pandas as pd

from domain.diamond import Diamond


class AssetClient:
    def __init__(self, asset_path: str):
        self.asset_path = asset_path
        self.asset_data = self._read_asset_data()

    def _read_asset_data(self):
        return pd.read_csv(self.asset_path)

    def find_similar_diamonds(self, diamond: Diamond, n=3):
        filtered_diamonds = self.asset_data[
            (self.asset_data['cut'] == diamond.cut) &
            (self.asset_data['color'] == diamond.color) &
            (self.asset_data['clarity'] == diamond.clarity)
        ]

        filtered_diamonds['carat_diff'] = (filtered_diamonds['carat'] - diamond.carat).abs()
        filtered_diamonds = filtered_diamonds.sort_values(by='carat_diff')

        top_n_diamonds_dicts = filtered_diamonds.head(n).to_dict(orient='records')

        top_n_diamonds = [Diamond.from_dict(diamond) for diamond in top_n_diamonds_dicts]

        return top_n_diamonds
