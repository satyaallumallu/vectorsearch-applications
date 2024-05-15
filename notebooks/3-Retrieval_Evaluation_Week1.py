#standard library imports
import sys
sys.path.append('../')

from typing import Any
import time
import os

# utilities
from tqdm import tqdm
from rich import print
from dotenv import load_dotenv, find_dotenv
env = load_dotenv(find_dotenv(), override=True)

from src.evaluation.retrieval_evaluation import calc_hit_rate_scores, calc_mrr_scores, record_results
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.database_utils import get_weaviate_client
from src.preprocessor.preprocessing import FileIO


data_path = '../data/golden_datasets/golden_256.json'

#################
##  START CODE ##
#################


### Load QA dataset
golden_dataset = FileIO.load_json(data_path)
print(golden_dataset)
### Instantiate Weaviate client and set Collection name
#read env vars from local .env file
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
model_path = 'sentence-transformers/all-MiniLM-L6-v2'
#print(api_key)
#instantiate client, 
retriever = WeaviateWCS(endpoint=url, api_key=api_key, model_name_or_path=model_path)
print(retriever._client.is_live())
collection_name = 'Huberman_minilm_256'


#################
##  END CODE   ##
#################

# should see 100 queries
print(f'Num queries in Golden Dataset: {len(golden_dataset["queries"])}')

def retrieval_evaluation(dataset: dict, 
                         collection_name: str, 
                         retriever: WeaviateWCS,
                         retrieve_limit: int=5,
                         chunk_size: int=256,
                         query_properties: list[str]=['content'],
                         return_properties: list[str]=['doc_id', 'content'],
                         dir_outpath: str='./eval_results',
                         include_miss_info: bool=False
                         ) -> dict[str, str|int|float]:
    '''
    Given a dataset and a retriever evaluate the performance of the retriever. Returns a dict of kw and vector
    hit rates and mrr scores. If inlude_miss_info is True, will also return a list of kw and vector responses 
    and their associated queries that did not return a hit, for deeper analysis. Text file with results output
    is automatically saved in the dir_outpath directory.

    Args:
    -----
    dataset: dict
        Dataset to be used for evaluation
    collection_name: str
        Name of Collection on Weaviate host to be used for retrieval
    retriever: WeaviateWCS
        WeaviateWCS object to be used for retrieval 
    retrieve_limit: int=5
        Number of documents to retrieve from Weaviate host, increasing this value too high 
        will artificially inflate the hit rate score of your retriever.
    chunk_size: int=256
        Number of tokens used to chunk text. This value is purely for results 
        recording purposes and does not affect results. 
    return_properties: list[str]=['doc_id', 'content']
        list of properties to be returned from Weaviate host for display in response
    dir_outpath: str='./eval_results'
        Directory path for saving results.  Directory will be created if it does not
        already exist. 
    include_miss_info: bool=False
        Option to include queries and their associated kw and vector response values
        for queries that are "total misses"
    '''

    results_dict = {'n':retrieve_limit, 
                    'Retriever': retriever.model_name_or_path, 
                    'chunk_size': chunk_size,
                    'query_props': query_properties,
                    'kw_hit_rate': 0,
                    'kw_mrr': 0,
                    'vector_hit_rate': 0,
                    'vector_mrr': 0,
                    'total_misses': 0,
                    'total_questions':0
                    }
    
    start = time.perf_counter()
    miss_info = []
    for query_id, q in tqdm(dataset['queries'].items(), 'Queries'):
        results_dict['total_questions'] += 1
        hit = False
        
        try:
            kw_response = retriever.keyword_search(request=q, collection_name=collection_name, query_properties=query_properties,
                                                   limit=retrieve_limit, return_properties=return_properties)
            vector_response = retriever.vector_search(request=q, collection_name=collection_name, 
                                                   limit=retrieve_limit, return_properties=return_properties)
            
            #collect doc_ids and position of doc_ids to check for document matches
            kw_doc_ids = {result['doc_id']:i for i, result in enumerate(kw_response, 1)}
            vector_doc_ids = {result['doc_id']:i for i, result in enumerate(vector_response, 1)}
            
            #extract doc_id for scoring purposes
            doc_id = dataset['relevant_docs'][query_id]
 
            #increment hit_rate counters and mrr scores
            if doc_id in kw_doc_ids:
                results_dict['kw_hit_rate'] += 1
                results_dict['kw_mrr'] += 1/kw_doc_ids[doc_id]
                hit = True
            if doc_id in vector_doc_ids:
                results_dict['vector_hit_rate'] += 1
                results_dict['vector_mrr'] += 1/vector_doc_ids[doc_id]
                hit = True
                
            # if no hits, let's capture that
            if not hit:
                results_dict['total_misses'] += 1
                miss_info.append({'query': q, 'kw_response': kw_response, 'vector_response': vector_response})
        except Exception as e:
            print(e)
            continue
    

    #use raw counts to calculate final scores
    calc_hit_rate_scores(results_dict, search_type=['kw', 'vector'])
    calc_mrr_scores(results_dict, search_type=['kw', 'vector'])
    
    end = time.perf_counter() - start
    print(f'Total Processing Time: {round(end/60, 2)} minutes')
    record_results(results_dict, chunk_size, dir_outpath=dir_outpath, as_text=True)
    
    if include_miss_info:
        return results_dict, miss_info
    return results_dict

#################
##  START CODE ##
#################
eval_results = retrieval_evaluation(golden_dataset, collection_name, retriever)
print(eval_results)
retriever._client.close()