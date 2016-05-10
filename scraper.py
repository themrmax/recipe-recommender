import requests
import pandas as pd
import re
import os
import boto3
import sys
import json
from selenium import webdriver

recipe_table = "recipe_recommender_recipes_test"
users_table = "recipe_recommender_users_test"

def delete_tables(recipe_table,users_table):
    client = boto3.client('dynamodb')
    if recipe_table in client.list_tables()['TableNames']:
        client.delete_table(TableName=recipe_table)

    if users_table in client.list_tables()['TableNames']:
        client.delete_table(TableName=users_table)


def create_tables(recipe_table,users_table):
    """create table if it doesn't exist
     setup dynamodb connection. make sure that your credentials are stored in your
    system's environment variables. Free tier for DynamoDB is 25 read and
    25 write units per month, and these tables are within that range."""

    client = boto3.client('dynamodb')

    if recipe_table not in client.list_tables()['TableNames']:
        ProvisionedThroughput = {"ReadCapacityUnits":6,"WriteCapacityUnits":6}
        KeySchema = [ {"AttributeName": "UserID", "KeyType":"HASH"}]
        AttributeDefinitions = [ {"AttributeName": "UserID", "AttributeType":"S"}]
        client.create_table(TableName=recipe_table,ProvisionedThroughput=ProvisionedThroughput,KeySchema=KeySchema,AttributeDefinitions=AttributeDefinitions)

    if users_table not in client.list_tables()['TableNames']:
        ProvisionedThroughput = {"ReadCapacityUnits":6,"WriteCapacityUnits":6}
        KeySchema = [ {"AttributeName": "UserID", "KeyType":"HASH"}]
        AttributeDefinitions = [ {"AttributeName": "UserID", "AttributeType":"S"}]
        client.create_table(TableName=users_table,ProvisionedThroughput=ProvisionedThroughput,KeySchema=KeySchema,AttributeDefinitions=AttributeDefinitions)

def dump_dynamodb_data():
    client = boto3.client('dynamodb')
    data = []
    r = client.scan(TableName=recipe_table)
    data += r['Items']
    while 'LastEvaluatedKey' in r.keys():
        r = client.scan(TableName=recipe_table, ExclusiveStartKey=r['LastEvaluatedKey'])
        data += r['Items']

    with open(recipe_table, "w") as f:
        for l in data:
            f.write(json.dumps(l) + '\n')

    data = []
    r = client.scan(TableName=users_table)
    data += r['Items']
    while 'LastEvaluatedKey' in r.keys():
        r = client.scan(TableName=users_table, ExclusiveStartKey=r['LastEvaluatedKey'])
        data += r['Items']
    with open(users_table, "w") as f:
        for l in data:
            f.write(json.dumps(l) + '\n')

def get_recipe_data(recipe_table):
    """Retrieve recipe data from DynamoDB. Need to paginate through results using
    LastEvaluatedKey, as DynamoDB enforces a maximum result set size"""
    client = boto3.client('dynamodb')
    data = []
    r = client.scan(TableName=recipe_table)
    data += r['Items']
    while 'LastEvaluatedKey' in r.keys():
        r = client.scan(TableName=recipe_table, ExclusiveStartKey=r['LastEvaluatedKey'])
        data += r['Items']
    recipeids = [u['RecipeID']['S'] for u in data]
    recipetitles = [eval(u['recipe_json']['S'])['recipe_title'] for u in data]
    user_ids = [eval(u['user_ids']['S']) if 'user_ids' in u.keys() else None for u in data]
    return pd.DataFrame({"recipetitle":recipetitles,"user_ids":user_ids}, index=recipeids)

def get_likes(users_table):
    """Retrieve like data from DynamoDB. Need to paginate through results using
    LastEvaluatedKey, as DynamoDB enforces a maximum result set size"""
    client = boto3.client('dynamodb')
    data = []
    r = client.scan(TableName=users_table)
    data += r['Items']
    while 'LastEvaluatedKey' in r.keys():
        r = client.scan(TableName=users_table, ExclusiveStartKey=r['LastEvaluatedKey'])
        data += r['Items']
    users = [u['UserID']['S'] for u in data]
    recipeboxes = [u['recipe_box']['S'] for u in data]
    return pd.Series(recipeboxes, index = users)

def get_recipes():
    client = boto3.client('dynamodb')
    #get links for all recipes by doing a "latest recipe" search from the home page with a very high page limit (this actually displays all recipes up to page 10 rather than just paging through them) and extracting all links.
    print("getting latest recipe IDs")
    txt = requests.get('http://cooking.nytimes.com/?latest-type=all&page=100').text
    recipe_ids = set(re.findall('/recipes/(\d+)',txt))
    assert len(recipe_ids) > 4000

    #get user-id's from recipe notes. Need to use a browser here (i.e. not just requests) because the user page links only appear after the Javascript is rendered
    #also parse the rest of the recipe data and save to DB for possilbe future usage.

    #first see what recipes are already in the DB and only scrape the difference
    recipe_ids_db = get_recipe_data().index

    driver = webdriver.Firefox()
    driver.set_page_load_timeout(10)
    for recipe_id in recipe_ids.difference(set(recipe_ids_db)):
        print("downloading recipe {}".format(recipe_id))
        try:
            driver.get("http://cooking.nytimes.com/recipes/"+recipe_id)
            notes = driver.find_elements_by_class_name('note-name')
            user_links = [n.get_attribute('href') for n in notes]
            user_ids = [re.search('\d+', l).group(0) for l in user_links]
            recipe_title = driver.find_element_by_class_name('recipe-title').text
            ingredients_elements = driver.find_elements_by_xpath('//*[@itemprop="recipeIngredient"]')
            ingredients = [{"ingredient_name":i.find_element_by_class_name('ingredient-name').text,"quantity":i.find_element_by_class_name('quantity').text} for i in ingredients_elements]
            instruction_elements = driver.find_elements_by_xpath('//*[@itemprop="recipeInstructions"]/li')
            instructions = [{"instruction_number":i, "instruction_step": s.text} for i,s in enumerate(instruction_elements)]
            recipe_json= json.dumps({"recipeid":recipe_id, "recipe_title":recipe_title, "ingredients":ingredients, "instructions":instructions})
            client.put_item(TableName=recipe_table, Item={"RecipeID":{'S':recipe_id},"recipe_json":{"S":recipe_json}, "user_ids": {'S':str(user_ids)}})
        except Exception as E:
            print (E)

def get_recipe_titles(L):
    recipe_ids = set(L)
    client = boto3.client('dynamodb')
    recipe_ids_db = get_recipe_data().index
    for i, recipe_id in enumerate(recipe_ids.difference(set(recipe_ids_db))):
        try:
            print("downloading recipe {} of {}".format(i, len(recipe_ids)))
            txt = requests.get("http://cooking.nytimes.com/recipes/"+recipe_id).text
            recipe_title = re.search('<title>(.*) Recipe',txt).group(1)
            recipe_json= json.dumps({"recipeid":recipe_id, "recipe_title":recipe_title})
            print(recipe_title)
            client.put_item(TableName=recipe_table, Item={"RecipeID":{'S':recipe_id},"recipe_json":{"S":recipe_json}})
        except Exception as e:
            print(e)

def get_recipe_boxes():
    client = boto3.client('dynamodb')
    #visit each user's homepage, page down to a high number, and grab all recipe links -- these are the user's favourited recipes (no other recipe links should be present on this page). do a check that the number of recipe links matches the number quoted as their favourites.
    #first get the list of users from the recipes
    user_ids = set([i for j in get_recipe_data().user_ids for i in j])
    #now see which users are already in the DB
    user_ids_db = get_likes().index


    #get all the users's likes
    diff = user_ids.difference(set(user_ids_db))
    for i, user_id in enumerate(diff):
        print("downloading recipe box for user {}/{}".format(i,len(diff)))
        txt = requests.get('http://cooking.nytimes.com/'+user_id+'?page=10').text
        recipes = list(set(re.findall('/recipes/(\d+)',txt)))
        print("found {} recipes".format(len(recipes)))
        client.put_item(TableName=users_table, Item={"UserID":{'S':user_id},"recipe_box":{"S":str(recipes)}})

if __name__ == '__main__':
    if sys.argv[1] == 'create-tables':
        create_tables()
    elif sys.argv[1] == 'get-recipes':
        get_recipes()
    elif sys.argv[1] == 'get-recipe-boxes':
        get_recipe_boxes()
    else:
        create_tables()
        get_recipes()
        get_recipe_boxes()
