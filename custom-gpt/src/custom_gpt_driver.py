import json
import os
import sys

from custom_gpt import CustomGPT



if __name__ == '__main__':
        try:
            cgpt = CustomGPT()
            cgpt.execute(sys.argv)
        except Exception as e:
            print(f'The following exception has occurred: {e}')


