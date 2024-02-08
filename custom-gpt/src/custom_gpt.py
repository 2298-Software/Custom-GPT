import json
import logging
import os
from typing import List

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, \
    ServiceContext, Document, StorageContext, load_index_from_storage


class CustomGPT(object):

    def __init__(self):
        logging.basicConfig()
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Program is starting')
        self.model_path = None
        self.data_path = None
        self.training_documents = None
        self.model = None
        self.company_name = None

    def execute(self, args):
        if len(args) != 2:
            self.graceful_exit()

        conf_file_path = args[1]
        self.log.info(f'Conf path is {conf_file_path}')
        if not os.path.exists(conf_file_path):
            self.graceful_exit()

        with open(conf_file_path) as f:
            data = f.read()
            conf = json.loads(data)

        self.company_name = conf['company_name']
        self.model_path = conf['model_path']
        self.data_path = conf['data_path']

        while True:
            mode = input(f"Welcome to {self.company_name}'s GPT library!\n "
                         "\n1: Train New or Refresh Existing GPT"
                         f"\n2: Chat with the {self.company_name} GPT"
                         "\nExit: Exit Program"
                         "\n\nPlease choose an option: ")

            if mode == '1':
                self.create_model()
            elif mode == '2':
                self.load_model()
                self.query_model()
            elif mode == 'Exit':
                print('Goodbye!')
                exit(0)
            else:
                print('Please chose a list option.')
            print('\n\n')

    def graceful_exit(self):
        self.log.info(f'Please provide the path to a configuration file as an argument to the program! ')
        exit(1)

    def load_training_data(self):
        self.training_documents: List[Document] = SimpleDirectoryReader(self.data_path).load_data()
        self.log.info('training data has been loaded')

    def create_model(self):
        self.load_training_data()
        service_context = ServiceContext.from_defaults()
        self.model = GPTVectorStoreIndex.from_documents(documents=self.training_documents,
                                                        service_context=service_context,
                                                        show_progress=True)
        self.save_model()
        self.log.info('model has been created')

    def save_model(self):
        self.model.storage_context.persist(persist_dir=self.model_path)
        self.log.info('model has been saved')

    def load_model(self):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.model_path)
            self.model = load_index_from_storage(storage_context)
            self.log.info('model has been loaded')
        except FileNotFoundError:
            self.log.info(
                f'You are attempting to query a model but a model does not exist in the path you provided: {self.model_path}')

    def query_model(self):
        query_engine = self.model.as_query_engine()
        while True:
            prompt = input("Please provide a prompt/query/question for the GPT: ")
            response = query_engine.query(prompt)
            print(f'{response}\n')
