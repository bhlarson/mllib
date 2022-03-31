#!/usr/bin/python3

import sys
import os
import requests
import random
import json
from datetime import datetime, timedelta

sys.path.append('.')
sys.path.append('./workflow')
from utils.s3 import s3store, Connect
from utils.jsonutil import ReadDict, Dict2Json

# paraemters is a dictionary of parameters to set
def set_parameters(workflow, new_parameters, template_name='train'):
    parameters = workflow['workflow']['spec']['arguments']['parameters']
    for parameter in parameters:
        for key, value in new_parameters.items():
            if key == parameter['name']:
                parameter['value'] = value

    '''
    templates = workflow['workflow']['spec']['templates']
    template = next(filter(lambda d: d.get('name') == template_name, templates), None)
    if template:
        parameters = template['inputs']['parameters']
        for parameter in parameters:
            name = parameter['name']
            if name in new_parameters:
                parameter['value'] = new_parameters[name]
    '''


def run(workflow, argocreds):
    session = requests.session()

    workflowstr = '{}://{}/api/v1/workflows/{}'.format(
        'https' if argocreds['tls'] else 'http',
        argocreds['address'],
        argocreds['namespace'])

    tasks_resp = session.post(workflowstr, json=workflow, verify = False)
    print('url: {} \nstatus_code: {} \nresponse: {}'.format(tasks_resp.url, tasks_resp.status_code, tasks_resp.text))
    return 0 if tasks_resp.ok == True else -1 


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('--debug', '-d', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-test', action='store_true', help='Run unit tests')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-objectserver', type=str, default='store', help='Object server name.')

    parser.add_argument('--server', '-s', type=str, default=None, help='Argo Server.')

    parser.add_argument('--run', '-r', type=str, default='workflow/train_dev.yaml', help='Run workflow')
    parser.add_argument('--params', '-p', type=json.loads, default=None, help='Parameteers parsed by set_parameters  e.g.: {"output_name":"20220330h", "target_structure":0.0, "batch_size":3}')

    args = parser.parse_args()
    return args

def Test(args):

    s3, creds, s3def = Connect(args.credentails, s3_name=args.objectserver)
    if not s3:
        print("Failed to connect to s3 {} name {} ".format(args.credentails, args.objectserver))
        return -1

    argocreds = None
    if 'argo' in creds:
        if args.server is not None:
            argocreds = next(filter(lambda d: d.get('name') == args.server, creds['argo']), None)
        else:
            #argocreds = random.choice(creds['argo'])
            argocreds = creds['argo'][0]

    if not argocreds:
        print("Failed to find argo credentials for {}".format(args.server))
        return -1

    workflow = ReadDict(args.run)
    if not workflow:
        print('Failed to read {}'.format(args.run))
        return -1
    if args.params is not None and len(args.params) > 0:
        set_parameters(workflow, args.params)
    result = run(workflow, argocreds)

    return result


if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        debugpy.listen(address=(args.debug_address, args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    result = Test(args)
    sys.exit(result)