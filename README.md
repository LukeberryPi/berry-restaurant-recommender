## Challenge

The attached dataset contains information about customers and restaurant ratings. The task is open-ended and an opportunity to showcase your strengths. You can use any programming language and any tool to showcase your results.

What insights can you gather? How can you visualise the data? Examples of things you could investigate: Is there a difference in cuisine preference between males and females? Different age groups?

Also think about how you could use a Machine Learning algorithm to predict the ratings of the customers in the dataset, and how you would use the dataset to evaluate your model.


## Dataset

A ~16k row `restaurants.csv` file with the following shape:

```
     age  gender    budget     price               cuisine_type        rating
0  35_49  female  under_20  20_to_30     Latin American/Mexican     very good
1  35_49    male  under_20  30_to_50                   American     very good
2  50_64    male  under_20  30_to_50                      Asian       dislike
3  20_34  female  under_20  20_to_30                  Bars/Pubs     very good
4  50_64    male  under_20  under_20  Deli/Sandwiches/Fast Food  satisfactory
```

The unique values for each column are:

| age          | '35_49', '50_64', '20_34', 'under_19', '65_and_over'
| gender       | 'female', 'male'
| budget       | 'under_20', '20_to_30', 'over_50', '30_to_50'
| price        | '20_to_30', '30_to_50', 'under_20', 'over_50'
| cuisine_type | 'Latin American/Mexican', 'American', 'Asian', 'Bars/Pubs', 'Deli/Sandwiches/Fast Food', 'Continental', 'African', 'Breakfast/Brunch', 'Seafood', 'Mediterranean', 'Vegetarian/vegan', 'Cafe'
| rating       | 'very good', 'dislike', 'satisfactory', 'excellent'


## Techbologies

This project uses Python, uv, pandas, matplotlib, seaborn, and scikit-learn.

## Running locally

- Clone this repository
- Run the command `uv run main.py` in the root directory

## Output

This program will output the following:

```
test accuracy: a float type between 0 and 1, representing the accuracy of the model
predictions: an array of rating predictions depending on the sample data passed to the model
```

For example, the following sample data:

```
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
```

Will output:

```
test accuracy: 0.9352356524527092
predictions: [0 2]
```

Meaning that the model predicted the ratings for the sample data to be 'dislike' and 'very good' respectively, with ~93% accuracy.


## Exploratory Data Analysis

### Cuisine Preference by Gender

In the graph grid "gender_cuisine_preference.png", each graph represents the distribution between max_ratings (customers who rated a restaurant "excellent") and min_ratings (customers who rated a restaurant "dislike") between genders for each cuisine type.

A few considerations:

- African Cuisine: Men tend to appreciate this cuisine more than women, being that no woman gave an African Restaurant the maximum rating and the ratio of women rating as "dislike" is double the ratio of men rating "dislike".

- Bars & Pubs: Men also tend to appreciate this cuisine more than women. This may be related to the environment and also the food choices. The data supports this as it shows that men rate this cuisine type as "excellent" twice as much as women, with about half the "dislikes".

- Vegetarian & Vegan: Women showed a greater appreciation for this cuisine, given that they rate it as "excellent" over 3 times more often than men. The ratio of men rating it as "dislike" is also twice as high as their female counterpart.

- American & Breakfast/Brunch: Both genders rated American and Breakfast/Brunch as excellent a majority of the time, with no records of "dislike" ratings. This could support a hypothesis of a sampling bias.


### Price/Budget Ratio Impact on Ratings

I had the hypothesis that if a given customer's budget was lower than the price of the chosen restaurant, that would impact the ratings negatively. However, by observing the graph "price_budget_ratio_preference.png" The average price over budget ratio of max ratings is very similar to min_ratings, showing that this hypothesis is not supported.


## If I had more time I would've...

- Taken more care about the generated graphs, as the readability isn't great;
- Created modules for better code organization (e.g., 'EDA' and 'model' classes);
- Created a website that recommends a restaurant based on user input;
- Uploaded the file to S3 to show that I know how to do it;
- Explored Random Forest and XGBoost to see how the results differ;