import argparse
import os
import yaml
from fnmatch import fnmatch

from kubernetes import client
from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-uninstall', action='store_true',help='uninstall')
    parser.add_argument('-namespace', type=str, default="mllib", help='Install namespace')

    args = parser.parse_args()
    return args

def main(args):
    print('setup.py main({})'.format(args))
    os.system('echo pwd = $(pwd)')

    config_files = [
        'ns.yaml',
        'minio.yaml',
    ]

    if not args.uninstall:
        os.system("kubectl create secret tls tls-mllib -n {} --key ~/keys/privkey.pem --cert ~/keys/cert.pem".format(args.namespace))
    else:
        os.system("kubectl delete secret tls-mllib -n {}".format(args.namespace))

    for i, config_file in enumerate(config_files):
        if args.uninstall:
            cmd = 'microk8s.kubectl delete -f k8s/{}'.format(config_files[-1*i - 1])
        else:
            cmd = 'microk8s.kubectl apply -f k8s/{}'.format(config_file)

        print(cmd)
        os.system(cmd)


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