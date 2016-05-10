import boto3
from ast import literal_eval
import json
from flask import Flask, request
from recommender import *
import pandas as pd
import sys


app = Flask(__name__)

@app.route('/refreshPredictions', methods=['GET'])
def refresh_predictions():
    global similarities
    global recommender
    users_table = sys.argv[1] #"recipe_recommender_users_test"
    recipes_table = sys.argv[2] #"recipe_recommender_recipes_test"

    try:
        users_table = sys.argv[1] #"recipe_recommender_users_test"
        recipes_table = sys.argv[2] #"recipe_recommender_recipes_test"


        print("Initialising recommender, calculating item similarities ... ")
        print (users_table)
        df,likes,recipenames,vectors = load_data(users_table)
        # vectors = pd.read_csv('vectors_top_100_recipes.csv',index_col=0)
        print("Caluclating similarities...")
        if len(vectors)>0:
            print("setting recommender")
            similarities = get_item_similarities(vectors)
            recommender = lambda u : item_recommender(u, similarities)
        print("Initialisation Complete")
        return  '{"status": "initialised predictor"}'
    except Exception as e:
        print(e)
        recommender = None
        similarities = []
        print("Error Initialisation")
        return  '{"error": "no predictor initialised"}'

@app.route('/addData', methods=['GET'])
def add_data():
    users_table = sys.argv[1] #"recipe_recommender_users_test"
    recipes_table = sys.argv[2] #"recipe_recommender_recipes_test"

    try:
        #user literal_eval for better security
        user_id = str(literal_eval(request.args.get('user_id')))
        recipe_ids = literal_eval(request.args.get('recipe_ids'))
        client = boto3.client('dynamodb')
        maybe_recipe_box = client.get_item(TableName=users_table,Key={'UserID':{'S':user_id}})
        recipe_box = []
        if 'Items' in maybe_recipe_box.keys():
            recipe_box = eval(maybe_recipe_box['Item']['recipe_box'])
        else:
            recipe_box = []
        new_recipe_box = list(set(recipe_box).union(set(recipe_ids)))
        print(new_recipe_box)
        client.put_item(TableName=users_table, Item={"UserID":{'S':user_id},"recipe_box":{"S":str(new_recipe_box)}})
        return  '{"status": "added data"}'
    except Exception as e:
        print(e)
        return '{"status": "error, bad request"}'

@app.route('/predictInterests', methods=['GET'])
def predict_interests():
    global recommender
    if recommender == None:
        return '{"error": "recommender not initialised"}'

    users_table = sys.argv[1] #"recipe_recommender_users_test"
    recipes_table = sys.argv[2] #"recipe_recommender_recipes_test"

    try:
        user_id = str(literal_eval(request.args.get('user_id')))
        client = boto3.client('dynamodb')
        maybe_recipe_box = client.get_item(TableName=users_table,Key={'UserID':{'S':user_id}})
        if 'Item' in maybe_recipe_box.keys():
            recipe_box = json.loads(maybe_recipe_box['Item']['recipe_box']['S'].replace("\'","\""))
            recs = recommender(recipe_box)[0:5]
            return str(recs)
        else:
            return {"error": "No likes found for this userid"}
    except Exception as e:
        print(e)
        return '{"status": "error"}'

@app.route('/test')
def index():
    return '{"status": "OK"}'

if __name__ == '__main__':
    similarities = None
    recommender = None
    refresh_predictions()
    app.run(debug=True)
