'''
Script to convert dataset CSV to JSON
'''
import csv
import json
import os
import pandas as pd
import sklearn.feature_extraction 
import numpy

# Open the CSV file
with open('data/freelancer_job_postings.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    fieldnames = csvreader.fieldnames

    # Create a list to store dictionaries for each row
    data = []

    # Loop through each row in the CSV file
    for row in csvreader:
        data.append(row)

# Create the outer dictionary
output = {"jobs": []}

# Open the JSON file for writing
with open('data/freelancer.json', 'w') as jsonfile:
    i = 0
    for entry in data:
        json_entry = {fieldname: entry[fieldname] for fieldname in fieldnames}
        json_entry["projectId"] = i
        i += 1
        output["jobs"].append(json_entry)

    json.dump(output, jsonfile, indent=4)

with open('data/freelancer.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    jobs_df = pd.DataFrame(data['jobs'])

#Builds and returns an inverted index 
def createInvIndex():
    corpus = []
    for jd in jobs_df['job_description']:
        corpus.append(jd)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names_out()
    return X