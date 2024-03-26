import os
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast


def load_or_create_tfidf_matrix():
    try:
        with open("tfidf_matrix.pickle", "rb") as f:
            tfidf_matrix, vectorizer = pickle.load(f)
    except FileNotFoundError:
        vectorizer = TfidfVectorizer()
        corpus = []
        with open('data/freelancer.json', 'r') as jsonfile:
          data = json.load(jsonfile)
          jobs_df = pd.DataFrame(data['jobs'])
        for jd in jobs_df['job_description']:
            corpus.append(jd)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        with open("tfidf_matrix.pickle", "wb") as f:
            pickle.dump((tfidf_matrix, vectorizer), f)
    return tfidf_matrix, vectorizer

# Precompute the inverted index, save in a global variable
tfidf_matrix, vectorizer = load_or_create_tfidf_matrix()

def cosine_sim(query: str):
    # No need to create a new TfidfVectorizer, just use the precomputed one
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, tfidf_matrix).ravel()
    return sims

def get_top5(query: str, docs: list, jobs_df: pd.DataFrame):
    sims = cosine_sim(query)
    top_indices = sims.argsort()[-5:][::-1]
    top_sims = [(sims[i], jobs_df.iloc[i]['job_title'], jobs_df.iloc[i]['client_average_rating'], docs[i], ast.literal_eval(jobs_df.iloc[i]['tags']), jobs_df.iloc[i]['avg_price'], jobs_df.iloc[i]['currency'])
                for i in top_indices]
    return top_sims

# Store json into python dict
# data = None 
# with open('./data/freelancer.json', 'r') as json_file:
#   data = json.load(json_file)

# jobs_df = pd.DataFrame(data['jobs'])
# jobs_desc = jobs_df['job_description'].values

# query = "Your client is a meal delivery company that operates in multiple cities. They have various fulfilment centres in these cities for dispatching meal orders to their customers. The client wants to help these centres with demand forecasting for upcoming weeks so that these centres will plan the stock of raw materials accordingly. The replenishment of most raw materials is done on a weekly basis and since the raw material is perishable, procurement planning is of utmost importance. Secondly, staffing of the centres is also one area wherein accurate demand forecasts are really helpful. We have the below information with us in the form of 3 different datasets: o Historical data of demand for a product-centre combination o Product (Meal) features such as category, sub-category, current price, and discount o Information for fulfilment centres like centre area, city information, etc. The dataset required has been provided along with this document. You need to come up with a story in Tableau that talks about the level of demand in each centre. This analysis needs to be granular enough to include product information as well. The client wants an end-to-end report to understand which fulfilment areas are doing well and which aren't. You can also talk about centre-meal combinations to add nuance to your final  "

# top5 = get_top5(query, jobs_desc)

# for idx, desc in enumerate(top5):
#   print(idx, desc)

