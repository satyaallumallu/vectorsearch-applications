# load libraries and functions
import sys
sys.path.append('../')

from dotenv import load_dotenv, find_dotenv
envs = load_dotenv(find_dotenv(), override=True)

from warnings import filterwarnings
filterwarnings('ignore')

from src.database.database_utils import get_weaviate_client
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.llm.llm_interface import LLM
from src.llm.llm_utils import get_token_count
from src.llm.prompt_templates import (huberman_system_message, 
                                      question_answering_prompt_series,
                                      generate_prompt_series)
from src.preprocessor.preprocessing import FileIO
from src.reranker import ReRanker
from litellm import ModelResponse

from typing import Literal
from rich import print
import os

## Set Constants
retriever = get_weaviate_client()
reranker = ReRanker()
collections = retriever.show_all_collections()

# define collection_name
print()
print(collections)
collection_name = 'Huberman_minilm_256'

# use question bank from previous notebook
queries = ['How to fight age-related muscle loss', 
           'How can listeners support the Huberman Lab podcast', 
           'What is the role of the mid-singulate cortex in the human brain', 
           'Why is sleep so important to maximizing healthspan']


# 1st-Stage Retrieval: Deep search over 200 documents
responses = [retriever.hybrid_search(q, collection_name, alpha=0.25, limit=200) for q in queries]

# 2nd-Stage Reranking: keeping only top-3 results
reranked = [reranker.rerank(resp, queries[i], top_k=3) for i,resp in enumerate(responses)]
print()
print(len(responses) == len(reranked))

# to see the actual prompt without numbers
print()
print(huberman_system_message)

# 1. We reiterate to the model what it's primary task is and we tell it what to expect given the format of the context.
# 2. We use a one-shot example of what the series of contexts will look like i.e. Summary, Guest, followed by Transcript chunk.
# 3. The **series** of context blocks (i.e. our retrieved results) is inserted here. Because we have useful 
# metadata (such as `summary` and `guest`) it makes sense to feed that to the LLM as well for additional context, 
# outside of just the transcript chunk (`content` field).
# 4. The original query is inserted here as the **question**: this is the end user information need.
# 5. We follow up with explicit directions on what to do after the LLM has reasoned through the text and reiterate 
# not to use external knowledge (i.e. model weights). We also add a **verbosity** option which effectively controls 
# the length of the model output (which can be adjusted based on our use case).

# print to see the prompt without numbers
print()
print(question_answering_prompt_series)

## Assistance Function: Generate a Series for User Prompt
user_prompt = generate_prompt_series(queries[0], reranked[0], verbosity_level=0)
print()
print(user_prompt)

## Check token counts
total_tokens = get_token_count([huberman_system_message, user_prompt])
print()
print(total_tokens)

## Putting it all together: Using the `litellm` library
print()
print(LLM.valid_models)

# instantiate the LLM Class
turbo = 'gpt-3.5-turbo-0125'
# the LLM Class will use the OPENAI_API_KEY env var as the default api_key 
llm = LLM(turbo)

from litellm import completion_cost

i = 0
#create new user prompt for each i value
user_prompt = generate_prompt_series(queries[i], reranked[i], verbosity_level=0)

response = llm.chat_completion(system_message=huberman_system_message,
                               user_message=user_prompt,
                               temperature=0.5,
                               stream=False,
                               raw_response=False)

print()
print(response)

# Compute estimated completion cost
cc = completion_cost(model=llm.model_name, 
                prompt=huberman_system_message+' '+user_prompt, 
                completion=response, 
                call_type='completion')
print(cc)

# Construct a basic RAG pipeline using the components that you have already built

from src.llm.prompt_templates import generate_prompt_series, create_context_blocks

class RAGPipeline:
    '''
    Basic RAG pipeline for exploratory purposes.
    '''
    def __init__(self, 
                 retriever: WeaviateWCS,
                 reader_llm: LLM, 
                 collection_name: str,
                 reranker: ReRanker=None,
                 system_message: str=huberman_system_message
                 ):
        
        #sensible components that won't change much from one run to the next
        self.retriever = retriever
        self.collection_name = collection_name
        self.reranker = reranker
        self.system_message = system_message
        self.llm = reader_llm

    #################
    ##  START CODE ##
    #################
    
    def __call__(self,
                 query: str,
                 alpha: float=0.25,
                 limit: int=200,
                 top_k: int=3,
                 verbosity: Literal[0,1,2]=0,
                 temperature: float=0.25,
                 max_tokens: int=500,
                 raw_response: bool=False,
                 **llm_kwargs
                 ) -> str | ModelResponse:
        '''
        Triggers retrieval, reranking, and LLM call. Returns LLM response.
        '''        
        search_results = self.retriever.hybrid_search(query, collection_name)
        if self.reranker:
            search_results = self.reranker.rerank(search_results, query, top_k)
        else:
            search_results = search_results[:top_k]
            
        user_message = generate_prompt_series(query, search_results)
        answer = self.llm.chat_completion(self.system_message, user_message, raw_response = True,
                                          **llm_kwargs)
        # completion_cost()
        
        retrieval_context = create_context_blocks(search_results)

        #results are returned as a dictionary for use later on in the LLM evaluation section
        return {'query': query, 'context': retrieval_context, 'answer': answer}

#instantiate your RAGPipeline Class
pipe = RAGPipeline(retriever, llm, collection_name, reranker, huberman_system_message)

#################
##  END CODE   ##
#################

## Test Functionality

output = pipe('How to fight age-related muscle loss', verbosity=1)
print()
print("LLM output is:")
# print(output['answer'])

# unpack data values
query, context, answer = [v for v in output.values()]

print('*'*100)
print(f'USER INPUT:\t{query}\n')
print('*'*100)
print(f'RETRIEVED CONTEXT:\n{context}\n\n')
print('*'*100)
print(f'MODEL RESPONSE: {answer}')

## Do We know that the LLM is effectively answering the user input based on the retrieved context?

from deepeval.test_case import LLMTestCase
from src.evaluation.llm_evaluation import AnswerCorrectnessMetric

## 1. Instantiate a metric by passing in the model name (string)
# metric takes in an evaluation LLM of our choice as a string
ac_metric = AnswerCorrectnessMetric('gpt-4-0125-preview')

# build a Test Case using previous vars
test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=context)

# should see a message from DeepEval 
ac_metric.measure(test_case)
print("Answer Correctness Metric: ")
print(ac_metric.__dict__)

## 5. A better way to view the metric data, to include the contextual data associated with the test 
# case is to use the `load_eval_response` function

from src.evaluation.llm_evaluation import load_eval_response 
eval_response = load_eval_response(ac_metric, test_case, return_context_data=True).to_dict()
print(eval_response)

fake_queries = [ "What are Musk's exact plans for taking over Mars", 
                 "What is Andrew Huberman's middle name", 
                 "How long do you think Peter Attia will live"
               ]

# this will make three calls to your designated reader model
packets = [pipe(q) for q in fake_queries]

# iterate over the packets and view the responses
responses = []
for packet in packets:
    
    #unpack each packet
    query, context, answer = [v for v in packet.values()]
    
    #create a test case
    test_case = LLMTestCase(query, answer, retrieval_context=context)
    
    #execute an AnswerCorrectness measure
    ac_metric.measure(test_case)
    
    #aggregate results
    responses.append(load_eval_response(ac_metric, test_case, return_context_data=True))

cost = sum([r.cost for r in responses])
scores = [r.score for r in responses]
reasons = [r.reason for r in responses]
print()
print("evaluation for the fake queries: ")
print(cost, scores, reasons)

# Assignment 2.4 - LLM Response Evaluation Baseline
""" # Option 1

from deepeval import evaluate
from src.evaluation.llm_evaluation import EvalResponse
from deepeval.test_case import LLMTestCase
from src.evaluation.llm_evaluation import AnswerCorrectnessMetric, load_eval_response
from notebook5_helpers import generate_project2_submission_file

def baseline_evaluation(test_cases: list[LLMTestCase],
                        evaluation_llm: str,
                        raw_response: bool=False
                        ) -> list[EvalResponse]:
    '''
    Execute bulk evaluation of test cases with a defined metric.
    '''
    #################
    ##  START CODE ##
    #################
    
    ac_metric = AnswerCorrectnessMetric(evaluation_llm)
    responses = evaluate(test_cases, [ac_metric], print_results=False)

    #################
    ##  END CODE   ##
    #################
    if raw_response:
        return responses
    eval_responses = [load_eval_response(r.metrics, r) for r in responses]
    
    scores = [r.score for r in eval_responses]
    eval_score = round(sum(scores)/len(scores),3)
    cost = [r.cost for r in eval_responses if r.cost]
    cost = sum(cost) if any(cost) else 'N/A'
    if cost == 'N/A':
        print(f'Total cost for this evaluation is not available. Non-OpenAI model cost information is not supported at this time.')  
    else: print(f'Total cost for this evaluation run: ${round(cost, 2)}')
    print(f'Evaluation Score: {eval_score}')
    
    return {'responses': eval_responses, 'scores': scores, 'evaluation_score': eval_score, 'cost': cost}


## Execute `baseline_evaluation`

#################
##  START CODE ##
#################

# 20 test cases have already been previously generated
test_case_data = FileIO.load_json('../data/golden_datasets/llm_eval_testcases_initial.json')

# have to be converted from dict to LLMTestCase format to work in evaluate function
baseline_test_cases = [LLMTestCase(**data) for data in test_case_data]

# execute evaluation
responses = baseline_evaluation(baseline_test_cases, 'gpt-4-0125-preview')

#################
##  END CODE   ##
#################

generate_project2_submission_file(responses) """

# Option 2

