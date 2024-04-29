'''
Script to convert dataset CSV to JSON
'''
import csv
import json
import os
import pandas as pd
import sklearn.feature_extraction 
import numpy
#from currency_converter import CurrencyConverter

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
# path = 'data/freelancer.json'

path = 'data/freelancer_with_USD.json'

'''with open(path, 'w') as jsonfile:
    i = 0
    c = CurrencyConverter()
    for entry in data:
        json_entry = {fieldname: entry[fieldname] for fieldname in fieldnames}
        json_entry['usd_val'] = 0
        json_entry["projectId"] = i
        usd_val = c.convert(json_entry['avg_price'], json_entry['currency'], 'USD')
        json_entry['usd_val'] = usd_val
        i += 1
        output["jobs"].append(json_entry)

    json.dump(output, jsonfile, indent=4)
    '''

with open(path, 'r') as jsonfile:
    data = json.load(jsonfile)
    jobs_df = pd.DataFrame(data['jobs'])


def print_job_titles():
    for job in jobs_df['job_title']:
        print(job)

print_job_titles()


#Builds and returns an inverted index 
def createInvIndex():
    corpus = []
    for jd in jobs_df['job_description']:
        corpus.append(jd)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names_out()
    return X