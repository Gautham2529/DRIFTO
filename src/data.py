"""
src/data.py
-----------
Loads the Breast Cancer dataset from scikit-learn, converts it to a
pandas DataFrame, and splits it into train/test (reference) sets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

BUILTIN_DATASETS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
}


def infer_target_column(df: pd.DataFrame) -> str | None:
    """Return the most likely target column for a tabular dataset."""
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            unique_values = series.nunique(dropna=True)
            if 1 < unique_values <= min(20, len(series)):
                return column
        elif series.dtype == "object":
            unique_values = series.nunique(dropna=True)
            if 1 < unique_values <= min(20, len(series)):
                return column
    return None


def load_dataset(
    dataset_name: str = "breast_cancer",
    data_path: str | Path | None = None,
    target_column: str | None = None,
    return_metadata: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load a built-in sklearn dataset or a CSV file.

    When *data_path* is provided, the dataset is read from disk and the target
    column is removed from the features before returning. If *return_metadata* is
    True, the function also returns feature names and sorted unique class labels.
    """
    if data_path is not None:
        dataframe = pd.read_csv(data_path)
        if target_column is None:
            target_column = infer_target_column(dataframe)
        if target_column is None:
            raise ValueError(
                "Could not infer a target column. Please pass target_column explicitly."
            )
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in {data_path}.")
        y = dataframe[target_column].copy()
        X = dataframe.drop(columns=[target_column]).copy()
        feature_names = list(X.columns)
        class_names = [str(value) for value in sorted(pd.unique(y), key=lambda val: str(val))]
        if return_metadata:
            return X, y, feature_names, class_names
        return X, y

    loader = BUILTIN_DATASETS.get(dataset_name.lower())
    if loader is None:
        available = ", ".join(sorted(BUILTIN_DATASETS))
        raise ValueError(f"Unsupported built-in dataset '{dataset_name}'. Available: {available}")

    raw = loader(as_frame=True)
    X: pd.DataFrame = raw.data.copy()
    y: pd.Series = raw.target.copy()
    feature_names = list(X.columns)
    class_names = [str(value) for value in sorted(pd.unique(y), key=lambda val: str(val))]
    if return_metadata:
        return X, y, feature_names, class_names
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and labels into train and test (reference) sets.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full label vector.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Seed used by the random number generator.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y, feature_names, class_names = load_dataset(return_metadata=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Dataset loaded - total samples : {len(X)}")
    print(f"  Train size : {len(X_train)}")
    print(f"  Test  size : {len(X_test)}")
    print(f"  Features   : {X.shape[1]}")
    print(f"  Target     : {y.name}  (classes: {class_names})")
