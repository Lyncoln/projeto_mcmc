import pandas as pd
import numpy as np
from typing import List, Tuple


class DataHandler:
    def __init__(self) -> None:
        self.ibov_df = self._prepare_raw_dataset()
    
    def _prepare_raw_dataset(self) -> pd.DataFrame:
        df = pd.read_csv('data/ibov_prices.csv')
        df.loc[:, 'DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values(by='DATE')
        df = df.dropna(subset=['PX_LAST'])
        df.loc[:, 'daily_return'] = df['PX_LAST'].pct_change()

        df = df[['DATE', 'PX_LAST', 'daily_return']]
        df = df.rename(columns={'DATE': 'date', 'PX_LAST': 'closing_price'})
        
        # adicionando outras features de retorno, volatilidade, ...
        df.loc[:, 'weekly_return'] = df['closing_price'].pct_change(5)
        df.loc[:, 'monthly_return'] = df['closing_price'].pct_change(22)
        df.loc[:, 'volatility'] = df['daily_return'].rolling(22).std() * np.sqrt(252)
        df = df.dropna()

        return df
    
    def prepare_gaussian_feature(self, feature: str) -> np.array:
        X = self.ibov_df.loc[:, feature].values
        X = X.reshape(-1, 1)

        return X
    
    def prepare_multiple_gaussian_features(self, features: List) -> np.array:
        X = np.concatenate([self.prepare_gaussian_feature(feature) for feature in features], axis=1)
        return X

