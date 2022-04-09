import sys
import os
import requests
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(''))
from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDictJson


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')   

    args = parser.parse_args()
    return args

def Test(args):

     s3, creds, s3def = Connect(args.credentails)

    s = requests.session()
    login_json = {"username": "bhlarson", "password": "264_Unstable"}                                                                                                                                          
    login_resp = s.post('http://localhost:8080/api/v1/auth/login', json=login_json)        
    print('Login ok={}'.format(login_resp.ok))
    s.headers['X-CSRFToken']=login_resp.cookies['csrftoken']

    response_about = s.get("http://localhost:8080/api/v1/server/about")
    print(response_about.text)
    
    task_dfn = {
        "name": "LIT_API",
        "project_id": 1,
        "bug_tracker": "",
        "status": "annotation",
        "subset": "Train"
    }
                                                                                
    tasks_resp = s.post("http://localhost:8080/api/v1/tasks", json=task_dfn)
    print(tasks_resp.json())

    images = s3.ListObjects(s3def['sets']['dataset']['bucket'], 'data/acquisition/lit/test/2021-09-08_14:48:59.839744/')
    remote_files = {"image_quality": 100,'remote_files':[]}


    for image in images:
        test_url = s3.GetUrl(s3def['sets']['dataset']['bucket'], image, expires=timedelta(days=7))
        remote_files['remote_files'].append(test_url)

    #dataApiEp = API_ENDPOINT + '/14/data' #14 is task id
    result_data = s.post('http://localhost:8080/api/v1/tasks/{id}/data'.format(id=tasks_resp.json()['id']),json=remote_files,auth=('user','pwd'))
    print(result_data.json())
    

    annotation_dfn = {
        "name": "LIT_API",
        "project_id": 1,
        "bug_tracker": "",
        "status": "annotation",
        "subset": "Train"
    }
    #dataApiEp = API_ENDPOINT + '/14/data' #14 is task id
    result_data = s.get('http://localhost:8080/api/v1/tasks/{id}/annotations?format'.format(id=18),json=annotation_dfn,auth=('user','pwd'))
    print(result_data.json())



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

    Test(args)