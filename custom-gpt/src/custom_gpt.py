import json
import os
from typing import List

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, \
    ServiceContext, Document, StorageContext, load_index_from_storage, Response


class CustomGPT(object):

    def __init__(self):
        self.model_path = None
        self.data_path = None
        self.model = None
        self.company_name = None

    def execute(self, args):
        if len(args) != 2:
            self.graceful_exit()

        conf_file_path = args[1]
        print(f'Conf path is {conf_file_path}')
        if not os.path.exists(conf_file_path):
            self.graceful_exit()

        with open(conf_file_path) as f:
            data = f.read()
            conf = json.loads(data)

        self.company_name = conf['company_name']
        self.model_path = conf['model_path']
        self.data_path = conf['data_path']

        while True:
            mode = int(input(f"Welcome to {self.company_name}'s GPT library!\n "
                             "\0: Exit Program"
                             "\n1: Train new gpt"
                             "\n2: Refresh Existing Model"
                             "\n3: View available GPTs"
                             "\n4: Submit a query to a GPT"
                             "\n5: Load a model"
                             "\n\nPlease choose an option: "))

            print('\n\n')
            if mode in [1, 2]:
                self.create_model()
            elif mode == 4:
                self.load_model()
            elif mode == 5:
                if self.model is not None:
                    self.query_model()
                else:
                    print('A model has not been loaded. Please load model before attempting to submit a prompt!')
            elif mode == 3:
                self.list_models()
            elif mode == 0:
                print('Goodbye!')
                exit(0)
            else:
                print('selection option not available')
            print('\n\n')

    def graceful_exit(self):
        print(f'Please provide the path to a configuration file as an argument to the program! ')
        exit(1)

    def load_training_data(self) -> List[Document]:
        documents: List[Document] = SimpleDirectoryReader(self.data_path).load_data()
        return documents

    def create_model(self):
        documents = self.load_training_data()
        service_context = ServiceContext.from_defaults(chunk_size_limit=3000)
        self.model = GPTVectorStoreIndex.from_documents(documents=documents, service_context=service_context,
                                                        show_progress=True)

    def save_model(self, model: GPTVectorStoreIndex):
        model.storage_context.persist(persist_dir=self.model_path)

    def load_model(self):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.model_path)
            self.model = load_index_from_storage(storage_context)
        except FileNotFoundError:
            print(
                f'You are attempting to query a model but a model does not exi3st in the path you provided: {self.model_path}')

    def query_model(self) -> Response:
        query_engine = self.model.as_query_engine()
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
