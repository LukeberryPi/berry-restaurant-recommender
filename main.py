import pandas as pd

def create_df(file_path):
    return pd.read_csv(file_path)

def cleanse_df(df):
    has_na_values = df.isna().any(axis=None)
    if has_na_values:
        # drop or populate...
        pass

    # i had to decide on a range for over_50 to assign a median to
    # since we don't have numerical pricing, i chose the previous ∆ = 20
    pricing_range_map = {
        'under_20': '5_to_20',
        '20_to_30': '20_to_30',
        '30_to_50': '30_to_50',
        'over_50': '50_to_70'
    }

    pricing_median_map = {
        '10_to_20': 12.5,
        '20_to_30': 25,
        '30_to_50': 40,
        '50_to_70': 60
    }

    df['budget'] = df['budget'].map(pricing_range_map).map(pricing_median_map).map(float)
    df['price'] = df['price'].map(pricing_range_map).map(pricing_median_map).map(float)

    df['cuisine_type'] = df['cuisine_type'].str.replace(' ', '_').str.lower()
    df['rating'] = df['rating'].str.replace(' ', '_')

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
    print(clean_df.head(100))
    print("\n\n")

if __name__ == "__main__":
    main()
