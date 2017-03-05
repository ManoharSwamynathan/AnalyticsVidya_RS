### Problem Statement (Recommendation Challenge)

Competition Source URL(Analytics Vidya): https://datahack.analyticsvidhya.com/contest/mlware-2/

Understanding customers and their preferences is the holy grail for online businesses. Building a recommender system is one of the common ways to do so.

In this contest, you need to build a model that predicts a given user’s ratings (from 0 to 10 stars) for a given item based on past ratings on other items and/or other information. The problem of rating prediction is the primary part of a recommendation problem (the part where explicit ratings are given). No additional information (user demographics, item content features etc.) are given and the prediction has to be made using only the ratings of already rated items.


#### Dataset
The rating data of 40,000 users, and 120 items . Ratings of users who have rated less than 10 items have been removed.

training.csv - This contains 958,529 ratings which are selected randomly from 1,599,544 ratings. Contains 4 columns:
* ID - Unique ID for each record
* userId - Unique user ID for each customer
* itemid - Item ID fo the product
* rating - Rating given to each item by user

test.csv - This file has three columns containing the ID, userId and itemId. The predictions on this set would be judged.

#### Evaluation:

The metrics used for evaluating the performance of the model is the Root Mean Squared Error between the predicted and the actual ratings.

Public : Private leaderboard split on test data is 25:75 

#### Results
Model scored 2.727915 on the private leaderboard
