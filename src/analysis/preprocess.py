from pandas import DataFrame

class MinMaxScaler:
    factor = 1.5

    def __init__(self, df: DataFrame, feat: str):
        self.df = df
        self.feat = feat
        self.val_max = df[feat].max()
        self.val_min = df[feat].min()

    def transform(self, new_df: DataFrame):
        new_df[self.feat] = (new_df[self.feat] - self.val_min) / (MinMaxScaler.factor * self.val_max - self.val_min)

    def reverse_transform(self, new_df: DataFrame):
        new_df[self.feat] = new_df[self.feat] * (MinMaxScaler.factor * self.val_max - self.val_min) + self.val_min
