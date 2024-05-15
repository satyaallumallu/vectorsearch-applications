import sys
sys.path.append('../')

#load from local .env file
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#external files
from src.preprocessor.preprocessing import FileIO
from src.database.weaviate_interface_v4 import WeaviateIndexer, WeaviateWCS
from src.database.database_utils import get_weaviate_client

#standards
import os
import time
import json
from typing import List
from tqdm import tqdm
from rich import print  # nice library that provides improved printing output (overrides default print function)

#######
#read env vars from local .env file
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
model_path = 'sentence-transformers/all-MiniLM-L6-v2'
#print(api_key)
#instantiate client, 
client = WeaviateWCS(endpoint=url, api_key=api_key, model_name_or_path=model_path)
print(client._client.is_live())

# Alternatively you can simply use the get_weaviate_client 
# convenice function, which assumes a default config
# client = get_weaviate_client()

#### Load Data
data_path = '/Users/sallumallu/Desktop/Online_Courses/Uplimit/RAG/huberman-minilmL6-256.parquet'

data = FileIO.load_parquet(data_path)
print(data[0].keys())

####
from src.database.properties_template import properties
print(properties)

####
#create your own Collection name or use the example from above
collection_name = 'Huberman_minilm_256' # 'Huberman_minilm_256'

""" ####
client.create_collection(collection_name=collection_name, properties=properties, description='Huberman Labs: 193 full-length transcripts')
print(client.show_collection_config(collection_name))

####
indexer = WeaviateIndexer(client)
batch_object = indexer.batch_index_data(data, collection_name)
 """
total_docs = client.get_doc_count(collection_name)
print(total_docs)

#####
#get properties that are part of the class
display_properties = [prop.name for prop in client.show_collection_properties(collection_name)]

# for this example we don't want to see the 
# summary or keywords so remove them
display_properties.remove('summary')
display_properties.remove('keywords')
display_properties

query = "What can I do to increase both my healthspan and lifespan"

# keyword search
response = client.keyword_search(request=query,
                                 collection_name=collection_name,
                                 query_properties=['title', 'guest', 'content'],  # change these up or remove one or two and see how the results change
                                 limit=5,
                                 filter=None,       # filtering is discussed as an optional final part of this notebook
                                 return_properties=display_properties,
                                 return_raw=False)  # turn this flag on and off to see how the responses are being reformatted

print(response)

# vector search
vector_response = client.vector_search(request=query,
                                       collection_name=collection_name,
                                       limit=5, 
                                       return_properties=display_properties,
                                       filter=None,
                                       return_raw=False,
                                       device='cpu')
print(vector_response)
client._client.close()
# compare results
keyword_ids = [doc['doc_id'] for doc in response]
vector_ids = [doc['doc_id'] for doc in vector_response]

print(f'Keyword IDs: {keyword_ids}')
print(f'Vector IDs: {vector_ids}')