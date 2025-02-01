import pandas as pd

def create_df(file_path):
    return pd.read_csv(file_path)

def cleanse_df(df):
    has_na_values = df.isna().any(axis=None)
    if not has_na_values:
        return df

def explore_df(clean_df):
    # correlations...
    pass

def preprocess(df):
    # feature eng, make ordinal...
    pass

def train_model(x, y):
    #
    pass

def main():
    df = create_df("restaurants.csv")
    clean_df = cleanse_df(df)
    print(clean_df.head())

if __name__ == "__main__":
    main()
