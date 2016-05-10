# Overview
This is a recommener system for cooking.nytimes.com. There is a scraper to download user's favourited recipes, which is used to seed the recommender. The main app is a Flask webserver with two endpoints `/addData` and `/predictInterests`

# Installation

## AWS DynamoDB
This project uses AWS DynamoDB for its datastore, so before you start, create a user in AWS with read/write permissions for DyanamoDB, and set your environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` appropriately.

**Important:** Run all commands from inside the recommender directory, not the project root.

## Scraper

The scraper uses selenium with Firefox, to run install Firefox (tested with version 30.0, might not work with later versions.

To run the scraper, simply

    python scraper.py

This will automatically create the DynamoDB tables if they don't exist already, and write the scraped values to DynamoDB

## App

Run the webservice with

    python app.py <user_table_name> <recipes_table_name>

The app has three endpoints.

### addData
This is for adding an array of user likes, eg.

    curl http://127.0.0.1:5000/addData?user_id="321"&recipe_ids=["123", "456"]


### refreshPredictions
This is to update the item-similarity matrix used for making predictions:

    curl http://127.0.0.1:5000/refreshPredictions

### predictInterests
This will get the top 5 recommendations for a user based on item similarity with known users, since last running refreshPredictions

    curl http://127.0.0.1:5000/predictInterests?user_id=123

## Testing

### Doctests
Run doctests for the recommender with:

    python -m doctest -v recommender.py

### Integration test.

This will create a test DynamoDB table, upload a training sample of data (200 users restricted to 100 recipes scraped from the NYTimes website), and run recommender against test subset of these, calculating the Average Precision@5 compared to the baseline recommender which recommends the top 5 recipes.

Run the integration testing script as:

    python recommender_test.py
