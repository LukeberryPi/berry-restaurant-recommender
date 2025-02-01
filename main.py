import pandas as pd

def create_df(file_path):
    return pd.read_csv(file_path)

def cleanse_df(df):
    # either remove or populate with avg/median NA values
    pass

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
    print(df.head())

if __name__ == "__main__":
    main()
