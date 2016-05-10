import numpy as np
import random
import pandas as pd
import boto3
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scraper import get_recipe_data, get_likes
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.preprocessing import Imputer
from sklearn.metrics.pairwise import cosine_similarity



def report_precision(recommender, likes, train_size, n, random_state=None):
    """Given a recommender and a list of user likes, calculate precision usin the,
leave-one-out method: for each row in data, take a sample of k=train_size likes for training, make the recommendations, and test the recommendations on the remaining likes
    >>> recommender = lambda u: [4,1] if 1 in u else [3,2]
    >>> report_precision(recommender, [[1,4],[2,3]], 1, 2, random_state=1)
    0.5
    >>> report_precision(recommender, [[1,4],[2,4]], 1, 2, random_state=1)
    0.25
    """
    if random_state:
        random.seed(random_state)

    def score_row(recommender, u, train_size, n):
        train = random.sample(u,train_size)
        test = set(u).difference(set(train))
        recs = recommender(train)
        #remove recs that the user has already liked in the training set
        recs = [r for r in recs if r not in train]
        n_correct = len(set(recs[0:5]).intersection(set(test)))
        # print("N correct = {}".format(n_correct))
        return n_correct

    avg_precision = np.sum(score_row(recommender,u, train_size, n) for u in likes)/(n*len(likes))
    return avg_precision



def neibourhood_recommender(u, vectors):
    """
    Input is a list of likes (reipeids), recommendations are caluclated by taking
    the most popular recipes from the neibourhood consisting of people with
    at least n=n_in_common likes in common.
    >>> vectors = pd.DataFrame([[1,1,0,0],[1,1,0,0],[1,0,1,0],[0,0,1,1]], columns = list('abcd'))
    >>> neibourhood_recommender(['c'], vectors)
    ['d', 'a', 'b']
    """

    neibourhood_mask = vectors.loc[:,u].sum(axis=1) >= 1
    neibourhood = vectors.loc[neibourhood_mask,:]
    uvec = [c in u for c in vectors.columns]
    dist = cdist([uvec], neibourhood, metric='jaccard')[0]
    P = neibourhood.mul(1-dist, axis=0).head().sum() / neibourhood.sum()
    recs = P.sort_values(ascending=False).index
    return recs[~recs.isin(u)].tolist()

def train_gbm_models(vectors, train_index, test_index):
    to_train = vectors.columns[vectors.sum() > 300]
    models = {}
    models['columns'] = vectors.columns
    for i, target in enumerate(to_train):
        # print("Training model {} of {}".format(i, len(to_train)))
        vectors_train = vectors.drop(target,axis=1)
        model = Pipeline([('imputer', Imputer()),
                          ('pca', PCA(n_components=20)),
                          ('gbm', GradientBoostingClassifier(n_estimators=20))])
        model.fit(vectors_train.ix[train_index], vectors.ix[train_index][target])
        models[target] = model
    return models

def train_logit_models(vectors, train_index, test_index,n_to_train):
    to_train = vectors.sum().sort_values(ascending=False)[0:n_to_train].index
    models = {}
    models['columns'] = vectors.columns
    for i, target in enumerate(to_train):
        print("Training model {} of {}".format(i, len(to_train)))
        vectors_train = vectors.drop(target,axis=1)
        model = Pipeline([('imp', Imputer()),('pca', PCA(n_components=99)), ('lr', LogisticRegression())])
        model.fit(vectors_train.ix[train_index], vectors.ix[train_index][target])
        models[target] = model
    return models

def recommender_gbm(u, models):
    vec = pd.Series(index = models['columns'])
    vec[u] = 1
    # vec = vec.fillna(0)
    scores = [(m.predict_proba(vec.drop(i).reshape(1,-1))[0][1], i) for i,m in models.items() if i != 'columns']
    recs = [l[1] for l in sorted(scores, reverse=True)]
    return recs


def popularity_vector(vectors):
    """ Given a matrix of vectorized likes, return a list of the most popular items (columns)
    >>> vectors = pd.DataFrame([[1,0,0,0],[1,1,0,0],[1,0,1,0],[0,0,1,1]])
    >>> popularity_vector(vectors)
    [0, 2, 3, 1]
    """
    return vectors.sum().sort_values(ascending=False).index.tolist()

def get_item_similarities(vectors):
    """Given a dataframe of like vectors, return the pairwise cosine similarities
    of the items (columns)
    >>> vectors = pd.DataFrame([[1,1,0,0],[1,1,0,0],[1,0,1,0],[0,0,1,1]], columns = list('abcd'))
    >>> sims = get_item_similarities(vectors)
    >>> sims.loc['a','a'] - 1 < 0.0000001
    True
    >>> sims.loc['a','b'] - 2/np.sqrt(6) < 0.00000001
    True
    >>> sims.loc['c','d'] - 1/np.sqrt(2) < 0.00000001
    True
    """
    similarities = cosine_similarity(vectors.transpose())
    similarities = pd.DataFrame(similarities,index = vectors.columns, columns = vectors.columns)
    return similarities

def item_recommender(u, similarities):
    """item recommender based on a similarity matrix
    >>> sims = pd.DataFrame([[1,0,1],[0,1,0],[1,0,1]])
    >>> item_recommender([1],sims)
    [2, 0]
    """
    p = similarities[u].sum(axis=1).sort_values(ascending=False)
    return p.index[p.index.isin(u) == False].tolist()

def load_data(users_table, recipes_table=None):
    print("Getting user likes")
    df = get_likes(users_table)
    #filter out users with less than 10 likes
    df = df[df.map(lambda x: len(eval(x)) >= 10)]
    likes = df.map(lambda x: eval(x))
    print("Getting recipe data")
    #recipenames = get_recipe_data(recipes_table)
    recipenames = None
    print("Vectorising user data...")
    vectors = (df.astype('str')
                .str.replace("\[|\]|'| ",'')
                .str.get_dummies(sep=','))
    return df,likes,recipenames,vectors
if __name__ == '__main__':

    #load data
    df,likes,recipenames,vectors = load_data()

    #80/20 train-test split
    train_index = np.random.random(len(likes)) < 0.8
    test_index = ~train_index

    #baseline: just use popularity to recommend
    recs = popularity_vector(vectors[train_index])
    recommender = lambda u : recs
    report_precision(recommender, likes[test_index], 5, 5, random_state=1)

    #user similarity collaborative filter
    recommender = lambda u : neibourhood_recommender(u, vectors[train_index])
    report_precision(recommender, likes[test_index], 5, 5, random_state=1)

    #item based cosine similarity

    similarities = get_item_similarities(vectors[train_index])
    recommender = lambda u : item_recommender(u, similarities)
    report_precision(recommender, likes[test_index], 5, 5, random_state=1)

    #restrict to top 100 recipees
    vectors = vectors[vectors.sum().sort_values(ascending=False).index[0:100]].copy()
    vectors = vectors[vectors.sum(axis=1) > 10].copy()
    likes = likes[vectors.index].copy()
    likes = likes.map(lambda l: [i for i in l if i in vectors.columns])
    train_index = np.random.random(len(likes)) < 0.8
    test_index = ~train_index
    #Popularity with Logistic Regression on Principle Components
    models = train_logit_models(vectors, train_index, test_index, 100)
    # with open("model.pkl", "wb") as f:
    #     f.write(pickle.dumps(models))

    recommender = lambda u : recommender_gbm(u, models)
    report_precision(recommender, likes[test_index], 5, 5, random_state=1)

    #Popularity with GBM on Principle Components
    models = train_gbm_models(vectors, train_index, test_index)
    # with open("model.pkl", "wb") as f:
    #     f.write(pickle.dumps(models))

    recommender = lambda u : recommender_gbm(u, models)
    report_precision(recommender, likes[test_index], 5, 5, random_state=1)
