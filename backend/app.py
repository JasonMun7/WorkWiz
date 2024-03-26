import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import analysis

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'data/freelancer.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    jobs_df = pd.DataFrame(data['jobs'])

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = jobs_df
    matches = merged_df[merged_df['job_title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['job_title', 'job_description', 'client_average_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def cosine_search(query):
    # Use the get_top5 function from the analysis.py file
    top_matches = analysis.get_top5(query, jobs_df['job_description'].values.tolist(), jobs_df)

    # Create a DataFrame from the top matching jobs
    top_matches_df = pd.DataFrame(top_matches, columns=['cosine_similarity', 'job_title', 'client_average_rating', 'job_description', 'tags', 'avg_price', 'currency'])

    # Convert the DataFrame to JSON
    top_matches_json = top_matches_df.to_json(orient='records')
    return top_matches_json


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/jobs")
def jobs_search():
    text = request.args.get("title")
    # return json_search(text)
    return cosine_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)