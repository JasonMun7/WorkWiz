'''
Script to convert dataset CSV to JSON
'''
import csv
import json

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
    for entry in data:
        json_entry = {fieldname: entry[fieldname] for fieldname in fieldnames}
        output["jobs"].append(json_entry)

    json.dump(output, jsonfile, indent=4)
