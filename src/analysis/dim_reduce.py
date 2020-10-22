import pandas as pd
from sklearn.ensemble import AdaBoostClassifier


# from sklearn.random_projection import SparseRandomProjection


# Feature importance
def get_feature_importance(df: pd.DataFrame, feat: str, n: int = 5000):
    """Get the importance of each figure wrt the specified feature"""

    df = df.sample(n, axis=0)
    model = AdaBoostClassifier().fit(df.loc[:, df.columns != feat], df[feat])

    return model.feature_importances_

# # Feature extraction
# def get_projection(data, n):
#     return SparseRandomProjection(n_componentsint=n).fit_transform(data)
