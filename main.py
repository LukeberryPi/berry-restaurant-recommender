import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_df(file_path):
    return pd.read_csv(file_path)

def cleanse_df(df):
    has_na_values = df.isna().any(axis=None)
    if has_na_values:
        # drop or populate...
        pass

    df = df.rename(columns={'budget': 'budget_range', 'price': 'price_range', 'age': 'age_range'})

    # i had to decide on a range for over_50 to assign a median to
    # since we don't have numerical pricing, i chose the previous ∆ = 20
    pricing_map = {
        'under_20': '10_to_20',
        '20_to_30': '20_to_30',
        '30_to_50': '30_to_50',
        'over_50': '50_to_70'
    }

    df['budget_range'] = df['budget_range'].map(pricing_map)
    df['price_range'] = df['price_range'].map(pricing_map)

    df['cuisine_type'] = df['cuisine_type'].str.replace(' ', '_').str.lower()
    df['rating'] = df['rating'].str.replace(' ', '_')

    return df

def explore_df(df):
    # correlations...
    if not os.path.exists("exploratory_analysis"):
        os.makedirs("exploratory_analysis")

    # budget / price ratio and how it affects rating
    # male vs female ratings by cuisine tyope
    # age rating by cuisine type


def preprocess(df):
    # feature eng, make ordinal...
    pricing_average_map = {
        '10_to_20': 15,
        '20_to_30': 25,
        '30_to_50': 40,
        '50_to_70': 60
    }

    df['average_budget'] = df['budget_range'].map(pricing_average_map).map(float)
    df['average_price'] = df['price_range'].map(pricing_average_map).map(float)

    ordinal_rating_map = {
        'dislike': 0,
        'satisfactory': 1,
        'very_good': 2,
        'excellent': 3
    }

    df['rating_ordinal'] = df['rating'].map(ordinal_rating_map)

    df['price_over_budget'] = df['average_price'] / df['average_budget'].map(float)

    return df

def train_model(x, y):
    #
    pass

def main():
    df = create_df("restaurants.csv")
    clean_df = cleanse_df(df)
    preprocessed_df = preprocess(clean_df)
    explore_df(clean_df)
    print(preprocessed_df.head(100))
    print("\n\n")
    for col in df:
        print(col, df[col].unique())


if __name__ == "__main__":
    main()
