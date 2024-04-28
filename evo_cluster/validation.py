import pandas as pd

def validate_input(df, columns, min_clusters, max_clusters, min_features, max_features):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty DataFrame.")

        if columns:
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} not found in DataFrame.")
        else:
            columns = df.columns.tolist()

        if any(not pd.api.types.is_numeric_dtype(df[col]) for col in columns):
            raise ValueError("All columns must be a numeric dtype for clustering.")

        if min_clusters < 1 or max_clusters < 1 or min_features < 1 or max_features < 1:
            raise ValueError("Minimum and maximum values for clusters and features must be at least 1.")

        if min_clusters > max_clusters:
            raise ValueError("min_clusters cannot be greater than max_clusters.")

        if min_features > max_features:
            raise ValueError("min_features cannot be greater than max_features.")
        
        if max_features > len(columns):
            raise ValueError("max_features cannot be larger than the number of available features/columns.")
