from pandas import DataFrame

class MinMaxScaler:
    factor = 1.5

    def __init__(self, df: DataFrame, feat: str):
        self.df = df
        self.feat = feat
        self.val_max = df[feat].max()
        self.val_min = df[feat].min()

    def transform(self, new_df: DataFrame, new_name: str = None):
        feat = self.feat if new_name is None else new_name
        new_df[feat] = (new_df[feat] - self.val_min) / (MinMaxScaler.factor * self.val_max - self.val_min)

    def reverse_transform(self, new_df: DataFrame, new_name: str = None):
        feat = self.feat if new_name is None else new_name
        new_df[feat] = new_df[feat] * (MinMaxScaler.factor * self.val_max - self.val_min) + self.val_min
