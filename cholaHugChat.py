from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, LLMPredictor, load_index_from_storage

import logging
import sys
import os
import openai
import langchain
import torch

from llama_index import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import BaseCache, Cohere, LLMChain, OpenAI
from langchain.base_language import BaseLanguageModel
#from langchain.chat_models import ChatOpenAI
from llama_index.llm_predictor import HuggingFaceLLMPredictor
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.indices.prompt_helper import PromptHelper

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader('data').load_data()

index_name = "./hugsaved_index"

prompt="000"
endflow = "999"
# set number of output tokens
num_output = 150
# set maximum input size
max_input_size = 512
# set maximum chunk overlap
max_chunk_overlap = 0.50


# define LLM
#llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
#stablelm_predictor = HuggingFaceLLMPredictor( max_input_size=4096, max_new_tokens=256,query_wrapper_prompt=query_wrapper_prompt, tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b", model_name="StabilityAI/stablelm-tuned-alpha-3b", device_map="auto", stopping_ids=[50278, 50279, 50277, 1, 0], tokenizer_kwargs={"max_length": 4096}, model_kwargs={"offload_folder": "offload"})
#stablelm_predictor = HuggingFaceLLMPredictor( max_input_size=4096, max_new_tokens=256,tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b", model_name="StabilityAI/stablelm-tuned-alpha-3b", device_map="auto", tokenizer_kwargs={"max_length": 4096}, model_kwargs={"offload_folder": "offload"})
stablelm_predictor = HuggingFaceLLMPredictor( max_input_size=4096, max_new_tokens=256, tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b", model_name="StabilityAI/stablelm-tuned-alpha-3b", device_map="balanced", stopping_ids=[50278, 50279, 50277, 1, 0], tokenizer_kwargs={"max_length": 4096}, model_kwargs={"offload_folder": "offload", "torch_dtype":torch.bfloat16})

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# configure service context
#service_context = ServiceContext.from_defaults(llm_predictor=stablelm_predictor)
#service_context = ServiceContext.from_defaults(embed_model=embed_model)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(chunk_size=1024, llm_predictor=stablelm_predictor, embed_model=embed_model)

if os.path.exists(index_name):
    print("Loading pre-built index for the documents from index folder")
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context, prompt_helper=prompt_helper)
else:
    print("Building index for the documents in data folder")
    index = VectorStoreIndex.from_documents( documents, service_context=service_context)
    index.storage_context.persist(persist_dir=index_name)

#index = VectorStoreIndex.from_documents(documents)
# build index

#index.storage_context.persist()

#query_engine = index.as_query_engine(verbose=True)
query_engine = index.as_query_engine(retriever_mode="embedding", verbose=True, service_context=service_context, response_mode="compact")
#query_engine = index.as_query_engine(retriever_mode="embedding", verbose=True, service_context=service_context, streaming=True)

# while (prompt != endflow):
#     prompt = input("Enter your query (999 to end): ")
#     if (prompt != endflow):
#         #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#         #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#         response = query_engine.query(prompt)
#         print(response)
#         #response_stream = query_engine.query(prompt)
#         #response_stream.print_response_stream()
#     else:
#         print("Exiting")



