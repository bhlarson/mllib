import sys
import os
import requests
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store, Connect
from utils.jsonutil import ReadDictJson


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_listen', type=str, default='0.0.0.0', help='Default, accept any client')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')   

    args = parser.parse_args()
    return args

def Test(args):

    s3, creds, s3def = Connect(args.credentails)
    

    session = requests.session()
    #login_json = {"username": "username", "password": "password"}                                                                                                                                          
    #login_resp = s.post('http://localhost:30992/api/v1/auth/login', json=login_json)        
    #print('Login ok={}'.format(login_resp.ok))
    #s.headers['X-CSRFToken']=login_resp.cookies['csrftoken']

    response = session.get("https://ipc001:30992/api/v1/info", verify = False)
    print(response.text)
    response = session.get("https://ipc001:30992/api/v1/userinfo", verify = False)
    print(response.text)
    response = session.get("https://ipc001:30992/api/v1/version", verify = False)
    print(response.text)
    response = session.get("https://ipc001:30992/api/v1/workflows/ml", verify = False)
    print(response.text)

    workflow = {
        "namespace": "argo",
        "serverDryRun": False,
        "workflow": {
            "metadata": {
                "generateName": "hello-world-",
                "namespace": "ml",
                "labels": {
                        "workflows.argoproj.io/completed": "false"
                    }
                },
            "spec": {
                "templates": [
                    {
                    "name": "whalesay",
                    "arguments": {},
                    "inputs": {},
                    "outputs": {},
                    "metadata": {},
                    "container": {
                    "name": "",
                    "image": "docker/whalesay:latest",
                    "command": [
                        "cowsay"
                    ],
                    "args": [
                        "hello world from argo workflow"
                    ],
                    "resources": {}
                    }
                }
                ],
                "entrypoint": "whalesay",
                "arguments": {}
            }
        }
    }

    tasks_resp = session.post("https://ipc001:30992/api/v1/workflows/ml", json=workflow)
    print(tasks_resp.json())
    
    task_dfn = {
        "name": "LIT_API",
        "project_id": 1,
        "bug_tracker": "",
        "status": "annotation",
        "subset": "Train"
    }
                                                                            



if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        import debugpy
        debugpy.listen(address=(args.debug_listen, args.debug_port))
        debugpy.wait_for_client()  # Pause until remote debugger is attached
        print("Debugger attached")

    Test(args)