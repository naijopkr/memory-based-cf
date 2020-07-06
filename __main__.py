import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('data/u.data', sep='\t', names=column_names)
df.head()

movie_titles = pd.read_csv('data/movie_id_titles.csv')
movie_titles.head()

df = pd.merge(df, movie_titles, on='item_id')
df.head()

n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()


# Memory-Based Collaborative Filtering

# Train test split
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size=0.25)

# Create two user-item matrices, one for training and another
# for testing
def generate_matrix(data: pd.DataFrame):
    matrix = np.zeros((n_users, n_items))
    for row in data.itertuples():
        """
        Row example:

        Pandas(Index=43275, user_id=494, item_id=286, rating=4,
        timestamp=879540508, title='English Patient, The (1996)')

            row[1]-1 == 494 - 1 == 493 >> user_id - 1 for index
            row[2]-1 == 286 - 1 == 285 >> item_id - 1 for column
            row[3] == 4 >> rating for value
        """
        index = row[1] - 1
        column = row[2] - 1
        rating = row[3]
        matrix[index, column] = rating

    return matrix

train_data_matrix = generate_matrix(train_data)
test_data_matrix = generate_matrix(test_data)

# Use pairwise distances to calculate the cosine similarity
from sklearn.metrics.pairwise import pairwise_distances

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    similarity_abs_sum = np.array([np.abs(similarity).sum(axis=1)])

    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = (
            mean_user_rating[:, np.newaxis] +
            similarity.dot(ratings_diff) /
            np.array([np.abs(similarity).sum(axis=1)]).T
        )
    elif type == 'item':
        pred = (ratings.dot(similarity) / similarity_abs_sum)

    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

# Evaluation with Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(y_true, y_pred):
    prediction = y_pred[y_true.nonzero()].flatten()
    ground_truth = y_true[y_true.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

user_rmse = rmse(test_data_matrix, user_prediction) # 3.1166
item_rmse = rmse(test_data_matrix, item_prediction) # 3.4446
