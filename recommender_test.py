import requests
import pandas as pd
import numpy as np
from recommender import popularity_vector, report_precision
import random
from scraper import create_tables, delete_tables
import time
import subprocess


add_data_url = "http://127.0.0.1:5000/addData?"
predict_interests_url = "http://127.0.0.1:5000/predictInterests?"
refresh_predictions_url = "http://127.0.0.1:5000/refreshPredictions"
data = pd.read_csv('vectors_top_100_recipes.csv', index_col=0)
data = data.iloc[200:400,:]


recipe_table = "recipe_recommender_recipes_test"
users_table = "recipe_recommender_users_test"

print('creating tables...')
create_tables(recipe_table,users_table)
time.sleep(15)
#80/20 train-test split
random.seed(1)
train_index = np.random.random(len(data)) < 0.8
test_index = ~train_index

print('starting webservice...')
p = subprocess.Popen(['python', 'app.py', "recipe_recommender_users_test", "recipe_recommender_recipes_test"])
time.sleep(10)

print("Loading training data into table")
for i,row in data[train_index].iterrows():
    print ("adding user {}".format(i))
    j = data.columns[data.loc[i,:]==1].tolist()
    r = requests.get(add_data_url + "user_id={}&recipe_ids={}".format(i,j))
    print (r.text)



print("refreshing predictions")
requests.get(refresh_predictions_url)
time.sleep(10)


print("Getting predictions for test data")
random.seed(1)
correct = []
for i,row in data[test_index].iterrows():
    print ("adding user {}".format(i))
    j = random.sample(data.columns[data.loc[i,:]==1].tolist(),5)
    r = requests.get(add_data_url + "user_id={}&recipe_ids={}".format(i,j)).text
    print ("testing user {}".format(i))
    r = requests.get(predict_interests_url + "user_id={}".format(i))
    n_correct = row[eval(r.text)].sum()
    print ("n_correct={}".format(n_correct))
    correct += [n_correct]


recs = popularity_vector(data[train_index])
recommender = lambda u : recs
l = data.columns
likes = data.apply(lambda x: l[x==1].tolist(),axis=1)
print ("Popularity benhmark precision: {}".format(report_precision(recommender, likes[test_index], 5, 5, random_state=1)))

print ("Recommender precision at 5 : {}".format(np.sum(correct)/5/sum(test_index)))

print ("Test complete")
print("killing webservice")
p.terminate()

print('deleting tables...')
delete_tables(recipe_table,users_table)
time.sleep(15)
