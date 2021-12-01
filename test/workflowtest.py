import math
import os
import sys
import copy
import io
import json


from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.abspath(''))

from utils.jsonutil import ReadDictJson
from utils.s3 import s3store, Connect


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    args = parser.parse_args()
    return args

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def main(args):
    print('Workflow Test')

    for i in tqdm(range(100)):
        print('{}'.format(i))

    print('Finished Workflow Test')


if __name__ == '__main__':
    import argparse
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        ''' https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        Launch application from console with -debug flag
        $ python3 train.py -debug
        "configurations": [
            {
                "name": "Python: Remote",
                "type": "python",
                "request": "attach",
                "port": 3000,
                "host": "localhost",
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "."
                    }
                ],
                "justMyCode": false
            },
            ...
        Connet to vscode "Python: Remote" configuration
        '''

        debugpy.listen(address=('0.0.0.0', args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    main(args)

