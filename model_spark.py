import pandas as pd
import numpy as np
import time

from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('favourites.csv').drop('userid',1)
# data = pd.DataFrame(np.random.random((10000,100))>0.5)


def model_i(data, i):
    model = GradientBoostingClassifier()
    model.fit(data.drop(i, 1), data[i])
    return model


top_recipes = data.sum().sort_values(ascending=False).index[0:1000]
var_index = sc.parallelize(top_recipes)
models = var_index.map(lambda x: model_i(data[top_recipes], x))
trained_models = models.collect()



def slow_function(x):
    time.sleep(60)

indexer = sc.parallelize(range(8))
result = indexer.map(lambda x: slow_function(x)).collect()
