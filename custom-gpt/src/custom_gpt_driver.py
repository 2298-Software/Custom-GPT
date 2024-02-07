import json
import os
import sys

from custom_gpt import CustomGPT


def main():
    print('\n\n')
    if mode in [1, 2]:
        cgpt.model = cgpt.create_model()
    elif mode == 4:
        cgpt.model = cgpt.load_model()
    elif mode == 5:
        if cgpt.model is not None:
            cgpt.query_model(cgpt.model)
        else:
            print('A model has not been loaded. Please load model before attempting to submit a prompt!')
    elif mode == 3:
        cgpt.list_models()
    elif mode == 0:
        print('Goodbye!')
        exit(0)
    else:
        print('selection option not available')
    print('\n\n')


def graceful_exit():
    print(f'Please provide the path to a configuration file as an argument to the program! ')
    exit(1)


if __name__ == '__main__':

    if len(sys.argv) !=2 :
        graceful_exit()

    conf_file_path = sys.argv[1]
    print(f'Conf path is {conf_file_path}')
    if not os.path.exists(conf_file_path):
        graceful_exit()

    with open(conf_file_path) as f:
        data = f.read()
        conf = json.loads(data)

    company_name = conf['company_name']
    model_path = conf['model_path']
    training_data_path = conf['data_path']
    cgpt = CustomGPT(model_path=model_path, data_path=training_data_path)

    while True:
        mode = int(input(f"Welcome to {company_name}'s GPT library!\n "
                         "\0: Exit Program"
                         "\n1: Train new gpt"
                         "\n2: Refresh Existing Model"
                         "\n3: View available GPTs"
                         "\n4: Submit a query to a GPT"
                         "\n5: Load a model"
                         "\n\nPlease choose an option: "))

        try:
            main()
        except Exception as e:
            print(f'The following exception has occurred: {e}')


