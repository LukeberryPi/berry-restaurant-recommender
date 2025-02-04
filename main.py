import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("restaurants.csv")

ordinal_rating_map = {
    'dislike': 0,
    'satisfactory': 1,
    'very_good': 2,
    'excellent': 3
}

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
    if not os.path.exists("exploratory_analysis"):
        os.makedirs("exploratory_analysis")

    # male vs female ratings by cuisine type
    df['max_rating'] = df['rating'] == 'excellent'
    df['min_rating'] = df['rating'] == 'dislike'

    gender_analysis = (
        df.groupby(['gender', 'cuisine_type'])
        .agg({'max_rating': 'mean', 'min_rating': 'mean'})
        .mul(100)
        .reset_index()
    )

    g = sns.FacetGrid(gender_analysis, col='cuisine_type', col_wrap=3, height=3, aspect=1)
    g.map_dataframe(
        # seaborn breaks if i remove **kwargs, so it has to remain
        lambda data, **kwargs: sns.barplot(
            x='gender',
            y='value',
            hue='variable',
            data=pd.melt(data, id_vars=['gender'], value_vars=['max_rating', 'min_rating']),
            dodge=True
        )
    )

    g.set_titles("{col_name}", size=10)
    g.set_axis_labels("", "% of ratings", size=10)
    g.add_legend(title="rating type", loc="center right", bbox_to_anchor=(1.25, 0.5))

    plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.4, right=0.8)
    g.figure.suptitle('extreme ratings by gender and cuisine', y=0.98, fontsize=14)
    plt.savefig("exploratory_analysis/gender_cuisine_preference.png", bbox_inches='tight')
    plt.clf()

    # calculate average price / budget ratio per rating
    ratings_order = ['dislike', 'satisfactory', 'very_good', 'excellent']
    avg_ratio = df.groupby('rating')['price_over_budget'].mean().reindex(ratings_order)

    plt.figure(figsize=(8,4))
    avg_ratio.plot.barh(color=["#CD5C5C", "#E0D26D", "#D8A26B", "#6B8E23"], edgecolor='black')

    plt.axvline(1.0, color='red', linestyle='--', linewidth=1)
    plt.text(1.05, 0.5, 'budget limit', color='black', va='center')

    plt.title('ratings by price/budget ratio')
    plt.xlabel('avg. price/budget ratio')
    plt.ylabel('')
    plt.xticks([0, 0.5, 1.0, 1.5])
    plt.yticks(ticks=range(4), labels=['dislike', 'satisfactory', 'very_good', 'excellent'])

    for i, v in enumerate(avg_ratio):
        plt.text(v + 0.05, i, f"{v:.2f}", va='center')

    plt.tight_layout()
    plt.savefig("exploratory_analysis/price_budget_ratio_preference.png")
    plt.clf()


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

    df['rating_ordinal'] = df['rating'].map(ordinal_rating_map)
    df['price_over_budget'] = df['average_price'] / df['average_budget'].map(float)

    return df

def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)
    model = GradientBoostingClassifier(random_state=8)
    model.fit(x_train, y_train)
    print("test accuracy:", model.score(x_test, y_test))
    return model

def predict_new(new_data, model):
    predictions = model.predict(new_data)
    print("predictions:", predictions)
    return predictions

def prepare_features(df):
    numeric = df[['average_budget', 'average_price', 'price_over_budget']]
    categorical = pd.get_dummies(df[['gender', 'age_range', 'budget_range', 'price_range', 'cuisine_type']], drop_first=True)
    features = pd.concat([numeric, categorical], axis=1)
    target = df['rating_ordinal']
    return features, target

def prepare_sample_features(sample_df, training_columns):
    numeric = sample_df[['average_budget', 'average_price', 'price_over_budget']]
    categorical = pd.get_dummies(sample_df[['gender', 'age_range', 'budget_range', 'price_range', 'cuisine_type']], drop_first=True)
    features = pd.concat([numeric, categorical], axis=1)
    features = features.reindex(columns=training_columns, fill_value=0)
    return features

def main():
    clean_df = cleanse_df(df)
    preprocessed_df = preprocess(clean_df)
    explore_df(clean_df)
    features, target = prepare_features(preprocessed_df)
    model = train_model(features, target)

    sample_df = pd.DataFrame({
        'gender': ['female', 'male'],
        'age_range': ['35_49', '20_34'],
        'budget_range': ['10_to_20', '20_to_30'],
        'price_range': ['20_to_30', '30_to_50'],
        'cuisine_type': ['deli/sandwiches/fast_food', 'asian'],
        'average_budget': [15, 25],
        'average_price': [25, 40],
        'price_over_budget': [1.66666667, 1.6]
    })
    sample_features = prepare_sample_features(sample_df, features.columns)
    predict_new(sample_features, model)
    return model

if __name__ == "__main__":
    main()
