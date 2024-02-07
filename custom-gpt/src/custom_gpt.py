import os
from typing import List

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, \
    ServiceContext, Document, StorageContext, load_index_from_storage, Response


class CustomGPT(object):

    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None

    def load_training_data(self, path_to_knowledge_dir: str) -> List[Document]:
        documents: List[Document] = SimpleDirectoryReader(path_to_knowledge_dir).load_data()
        return documents

    def create_model(self) -> GPTVectorStoreIndex:
        documents = self.load_training_data(self.data_path)
        service_context = ServiceContext.from_defaults(chunk_size_limit=3000)
        model = GPTVectorStoreIndex.from_documents(documents=documents, service_context=service_context,
                                                   show_progress=True)
        return model

    def save_model(self, model: GPTVectorStoreIndex):
        model.storage_context.persist(persist_dir=self.model_path)

    def load_model(self):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.model_path)
            self.model = load_index_from_storage(storage_context)
        except FileNotFoundError:
            print(
                f'You are attempting to query a model but a model does not exi3st in the path you provided: {self.model_path}')

    def query_model(self, model: GPTVectorStoreIndex) -> Response:
        query_engine = model.as_query_engine()
        while True:
            prompt = input("Please provide a prompt/query/question for the GPT: ")
            response = query_engine.query(prompt)
            return response

    def list_models(self):
        print(f'The following model(s) exist in path:\n')
        models = os.listdir(self.model_path)
        if len(models) == 0:
            print('Zero models found in the provided directory!')
        else:
            print(models)
