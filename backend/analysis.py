import os
import pandas as pd
import json
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(query : str, doc : str):
  vectorizer = TfidfVectorizer()

  # constructs tf idf matrix between query and doc 
  tfidf_mat = vectorizer.fit_transform([query, doc])

  # tf-idf of q 
  q = tfidf_mat[0]

  # tf-idf of doc
  doc = tfidf_mat[1]

  #cosine sim
  sim = cosine_similarity(q, doc)

  return sim[0][0]

def get_top5(query : str, docs : list):
  vectorizer = TfidfVectorizer()
  # constructs tf idf matrix between query and docs
  tfidf_mat = vectorizer.fit_transform([query] + docs)

  # tf-idf of q 
  q = tfidf_mat[0]

  n_docs = tfidf_mat.shape[0]
  sims = []
  for i in range(1, n_docs):
    doc = tfidf_mat[i]
    sim = cosine_similarity(q, doc)
    sims.append((sim[0][0], docs[i - 1]))
  sims.sort(reverse = True)
  return sims[:5]

# Store json into python dict
data = None 
with open('./data/freelancer.json', 'r') as json_file:
  data = json.load(json_file)

jobs_df = pd.DataFrame(data['jobs'])
jobs_desc = jobs_df['job_description'].values

query = "Your client is a meal delivery company that operates in multiple cities. They have various fulfilment centres in these cities for dispatching meal orders to their customers. The client wants to help these centres with demand forecasting for upcoming weeks so that these centres will plan the stock of raw materials accordingly. The replenishment of most raw materials is done on a weekly basis and since the raw material is perishable, procurement planning is of utmost importance. Secondly, staffing of the centres is also one area wherein accurate demand forecasts are really helpful. We have the below information with us in the form of 3 different datasets: o Historical data of demand for a product-centre combination o Product (Meal) features such as category, sub-category, current price, and discount o Information for fulfilment centres like centre area, city information, etc. The dataset required has been provided along with this document. You need to come up with a story in Tableau that talks about the level of demand in each centre. This analysis needs to be granular enough to include product information as well. The client wants an end-to-end report to understand which fulfilment areas are doing well and which aren't. You can also talk about centre-meal combinations to add nuance to your final  "

top5 = get_top5(query, jobs_desc)

for idx, desc in enumerate(top5):
  print(idx, desc)

