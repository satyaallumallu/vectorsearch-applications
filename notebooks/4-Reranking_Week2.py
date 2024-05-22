import sys
sys.path.append('../')

#load from local .env file
from dotenv import load_dotenv, find_dotenv
env = load_dotenv(find_dotenv(), override=True)

#standard python
from typing import List, Dict, Tuple
import os

# external libraries
from tqdm import tqdm
from rich import print  # nice library that provides improved printing output (overrides default print function)

# external files
from src.reranker import ReRanker
from src.database.weaviate_interface_v4 import WeaviateWCS

## Instantiate Weaviate Client
#read env vars from local .env file
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
model_path = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
#instantiate client
client = WeaviateWCS(url, api_key, model_name_or_path=model_path)

#display available collection names on cluster
print()
print("found following collections in your Weaviate datastore:")
print(client.show_all_collections())

#set collection name to run queries on
collection_name = 'Huberman_multiqaMiniLML6cosv1_256'

### Hybrid Search - `RelativeRankFusion
query = 'How will listening to Huberman Lab improve my life'
query_properties = ['title', 'guest', 'content']
kw_response = client.keyword_search(query, collection_name, query_properties, limit=10)
vector_response = client.vector_search(query, collection_name, limit=10)

# Extract scores and doc_id's
from collections import OrderedDict, defaultdict

def get_scores_ids(response: list[dict], include_cross_score: bool=False, limit: int=10):
    '''
    Extracts scores and ids from response object.
    If keyword, extracts from "score" field.
    If vector, extracts from "distance" field and 
    subtracts from 1 in keeping with the concept that
    higher scores are better. 
    '''
    def round_(score: str, place: int=3):
        return round(float(score),place)
        
    score_dict = OrderedDict()
    if response[0].get('score'):
        if include_cross_score: 
            return {d['doc_id'] : f'Score: {round_(d["score"])}   :   Cross-Score: {round_(d["cross_score"])}' for d in response[:limit]}
        return {d['doc_id'] : round_(d['score'], 3) for d in response[:limit]}
    return {d['doc_id'] : round_(1 - d['distance'],3) for d in response[:limit]}

# Raw scores

kw_scores = get_scores_ids(kw_response)
vec_scores = get_scores_ids(vector_response)
print()
print("Raw Scores:")
print(kw_scores, vec_scores)

# Normalize scores
# using sklearn to normalize scores
from sklearn.preprocessing import MinMaxScaler
from numpy import ndarray

scaler = MinMaxScaler()
kw_normal = scaler.fit_transform([[score] for score in kw_scores.values()])
vec_normal = scaler.fit_transform([[score] for score in vec_scores.values()])
print()
print("Normalized Scores:")
print(kw_normal, vec_normal)

# multiply by weighted alpha
alpha = 0.4
kw_weighted = kw_normal * (1 - alpha)
vec_weighted = vec_normal * alpha
print()
print("Weighted Scores:")
print(kw_weighted, vec_weighted)

# update final scores
def update_scores(weighted_kw_scores: ndarray, 
                  weighted_vec_scores: ndarray
                 ) -> None:
    for i, k in enumerate(kw_scores):
        kw_scores[k] = weighted_kw_scores[i][0]
    for i, k in enumerate(vec_scores):
        vec_scores[k] = weighted_vec_scores[i][0]

update_scores(kw_weighted, vec_weighted)

# the updated scores should look very different from their original raw scores
print()
print("Updated Scores:")
print(kw_scores, vec_scores)

# Add together any documents with the same doc_id and then sort final results
def add_doc_scores(kw_scores: Dict[str, float], 
                   vec_scores: Dict[str, float],
                   top_k: int=5
                  ) -> List[Tuple[str, float]]:
    '''
    Combined keyword and vector scores by adding values 
    for any duplicate docs and then sorts resulting 
    dictionary of results.  Returns top_k values. 
    '''
    hybrid_results = kw_scores
    for k,value in vec_scores.items():
        if k in kw_scores:
            hybrid_results[k] += value
        else:
            hybrid_results[k] = value
    return sorted(hybrid_results.items(), key=lambda x: x[1], reverse=True)[:top_k]

# compare these final ranked value with the original keyword and vector queries
ranked_results = add_doc_scores(kw_scores, vec_scores, top_k=5)
print()
print("Ranked Results:")
print(ranked_results)

# Compare manually generated scores with Weaviate hybrid method
# scores and ranking are slightly off due to the way that keyword results are processed
hyb_response = client.hybrid_search(query, collection_name, alpha=alpha, limit=5)
print()
print("Compare manually generated scores with Weaviate hybrid method:")
print(get_scores_ids(hyb_response))
print(ranked_results)


### Compare Search Methods
def print_results_by_key(results: list[dict], return_props: list[str]) -> None:
    '''
    Pretty-prints nested search results
    '''
    from rich.pretty import pprint
    keys = return_props + ['score', 'distance']
    for r in results:
        for key in r:
            if key in keys:
                pprint(f'{key.upper()}: {r[key]}')
        print('\n\n')
    print('-'*100)

def print_results(client: WeaviateWCS, 
                  collection_name: str, 
                  queries: list[str], 
                  return_props: list[str], 
                  alpha_value: float
                  ) -> None:
    '''
    Prints search results grouped by search method
    '''
    for q in queries:
        kw_result = client.keyword_search(q, collection_name, query_properties,return_properties=return_props, limit=3)
        vector_result = client.vector_search(q, collection_name, return_properties=return_props, limit=3)
        hybrid_result = client.hybrid_search(q, collection_name, query_properties, return_properties=return_props, alpha=alpha_value, limit=3) 
        print('*'*100)
        print(f'QUERY: {q}')
        print('*'*100)
        print(f'KEYWORD RESULTS:')
        print_results_by_key(kw_result, return_props)
        print(f'VECTOR RESULTS:')
        print_results_by_key(vector_result, return_props)
        print(f'HYBRID RESULTS:')
        print_results_by_key(hybrid_result, return_props)

queries = ['How to fight age-related muscle loss', 'How can listeners support the Huberman Lab podcast', 
'What is the role of the mid-singulate cortex in the human brain']
return_props = ['title', 'content', 'doc_id']
alpha_value = alpha

print_results(client, collection_name, queries, return_props, alpha_value)

## Instantiate ReRanker
# pass in model_id like any other HuggingFace model
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## Execute Search
# Going forward all of our searches will be hybrid with alpha = 0.25 and 
# we are increasing the `limit` to 200 to take advantage of the CrossEncoder

# re-initialize collection_name if needed 
# collection_name = ''

query = 'Why is sleep so important to maximizing healthspan'

# get search results, note that the limit value is set to 200
return_properties = ['guest', 'title', 'content', 'doc_id']
results = client.hybrid_search(query, collection_name, query_properties, return_properties=return_properties, limit=200, alpha=0.25)
print(get_scores_ids(results, include_cross_score=False))

## Rerank Results
print("top result before reranking step:")
print(results[0])

reranked = reranker.rerank(results, query, apply_sigmoid=True)
print("top result after reranking:")
print(reranked[0])

print(get_scores_ids(reranked, include_cross_score=True))

### Pause here and take a moment running different searches and comparing the results both before 
### and after the reranking step. Take all the time you need to convince yourself that a ReRanker 
### will be useful for the application you're building.

## Evaluation of Reranker Effect on Latency

import time
import pandas as pd

def time_search(client: WeaviateWCS, 
                collection_name: str,
                limit: int,
                rerank: bool
                ) -> float:
    '''
    Executes search given a limit value. 
    Returns total time in seconds
    '''
    query = 'What is the best long term strategy for fat loss'
    start = time.perf_counter()
    response = client.hybrid_search(query, collection_name, query_properties,limit=limit, return_properties=['content', 'title'])
    if rerank:
        reranked = reranker.rerank(response, query)
    end = time.perf_counter() - start
    return round(end, 3)

limit_values = list(range(10, 400, 10))

unranked_times = []
for n in tqdm(limit_values, 'Search: No Reranker'):
    unranked_times.append((time_search(client, collection_name, limit=n, rerank=False), n))

ranked_times = []
for n in tqdm(limit_values, 'Search: With Reranker'):
    ranked_times.append((time_search(client, collection_name, limit=n, rerank=True), n))

unranked = pd.DataFrame(unranked_times, columns=['time', 'n'])
ranked = pd.DataFrame(ranked_times, columns=['time', 'n'])

ax = unranked.plot.scatter(x='n', y='time', label='No Reranker', title='Result Size Effect on Latency')
ax2 = ranked.plot.scatter(x='n', y='time', ax=ax, color='orange', ylabel='Latency (ms)', label='With Reranker', xlabel='# of Returned Results')
hline = ax2.axhline(y = 0.5, color = 'r', linestyle = 'dashed', label = "Max allowable Latency")     
legend = ax2.legend(bbox_to_anchor = (1.0, 1)) 

## Reevaluate your retrieval results, this time using hybrid search and a Reranker
# Instructions:
# 1. Fill in the areas of the code wherever you see a `None` statement.  Use the `execute_evaluation` 
# function to run your retrieval benchmark.
# 2. `execute_evaluation` is the same function from Notebook 3, albeit with some modifications to 
# incorporate hybrid search and Reranker functionality.
# 3. Adjust the different hyperparameters to get the highest scoring retrieval score possible.

from src.evaluation.retrieval_evaluation import execute_evaluation
from src.preprocessor.preprocessing import FileIO

#################
##  START CODE ##
#################
path_golden_data_set = '/Users/sallumallu/Documents/GitHub/vectorsearch-applications/data/golden_datasets/golden_256.json'
golden_dataset = FileIO.load_json(path_golden_data_set)
retrieval_results = execute_evaluation(golden_dataset, collection_name, client, reranker, alpha_value, query_properties = query_properties)
print(retrieval_results)
#################
##  END CODE   ##
################# 