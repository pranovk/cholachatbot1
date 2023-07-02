from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, LLMPredictor, load_index_from_storage
import logging
import sys
import os
import openai
import langchain
from langchain import BaseCache, Cohere, LLMChain, OpenAI
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['OPENAI_API_KEY'] = "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"#"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"
openai.api_key = os.getenv('OPENAI_API_KEY')

documents = SimpleDirectoryReader('data').load_data()

index_name = "./Chola_policy_index"


prompt="000"
endflow = "999"

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# configure service context
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

if os.path.exists(index_name):
    print("Loading pre-built index for the documents from index folder")
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context)
else:
    print("Building index for the documents in data folder")
    index = VectorStoreIndex.from_documents( documents, service_context=service_context)
    index.storage_context.persist(persist_dir=index_name)

#index = VectorStoreIndex.from_documents(documents)
# build index

#index.storage_context.persist()

query_engine = index.as_query_engine(verbose=True)
#query_engine = index.as_query_engine()





