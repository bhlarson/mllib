import sys
import argparse
import json
import os
import subprocess
import shutil
from os import fspath
sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store
from zipfile import ZipFile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-path', type=str, default='/store/Datasets/cityscapes', help='coco dataset path')
parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-cityscapeurl', type=str, default='https://www.cityscapes-dataset.com', help='Cityscape URL')
parser.add_argument('-dataset', type=str, default='cityscapes', help='Dataset.')
parser.add_argument('-cookie', type=str, default='cookies.txt', help='Cookie file name')


citypackages=[
    {'id': 1, 'name': 'gtFine_trainvaltest.zip', 'size': '241MB'},
    {'id': 2, 'name': 'gtCoarse.zip', 'size': '1.3GB'},
    {'id': 3, 'name': 'leftImg8bit_trainvaltest.zip', 'size': '11GB'},
    {'id': 4, 'name': 'leftImg8bit_trainextra.zip', 'size': '44GB'},
    #{'id': 8, 'name': 'camera_trainvaltest.zip', 'size': '2MB'},
    #{'id': 9, 'name': 'camera_trainextra.zip', 'size': '8MB'},
    #{'id': 10, 'name': 'vehicle_trainvaltest.zip', 'size': '2MB'},
    #{'id': 11, 'name': 'vehicle_trainextra.zip', 'size': '7MB'},
    #{'id': 12, 'name': 'leftImg8bit_demoVideo.zip', 'size': '6.6GB'},
    #{'id': 28, 'name': 'gtBbox_cityPersons_trainval.zip', 'size': '2.2MB'},
]

def main(args):

    creds = {}
    with open(args.credentails) as json_file:
        creds = json.load(json_file)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))

    s3def = creds['s3'][0]
    s3 = s3store(s3def['address'], 
                 s3def['access key'], 
                 s3def['secret key'], 
                 tls=s3def['tls'], 
                 cert_verify=s3def['cert verify'], 
                 cert_path=s3def['cert path']
                 )

    os.makedirs(args.path, exist_ok=True)

    sysmsg = "wget --keep-session-cookies --save-cookies={} --post-data 'username={}&password={}&submit=Login' {}/login/".format(
        args.cookie,
        creds['cityscapes']['username'], 
        creds['cityscapes']['password'],
        args.cityscapeurl)
        
    #print(sysmsg)
    os.system(sysmsg)

    for citypackage in citypackages:
        #https://www.cityscapes-dataset.com/file-handling/?packageID=1
        url = 'wget --load-cookies {} --content-disposition {}/file-handling/?packageID={}'.format(
            args.cookie,
            args.cityscapeurl,
            citypackage['id'])
        outpath = '{}/{}'.format(args.path, citypackage['name'])
        if os.path.isfile(outpath):
            print('{} exists.  Skipping'.format(outpath))
        else:
            sysmsg = 'wget -O {} {} '.format(outpath, url)
            print(sysmsg)
            os.system(sysmsg)

        dest = '{}/{}'.format(args.path,args.dataset)
        with ZipFile(outpath,"r") as zip_ref:
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(member=file, path=fspath(dest))

        os.remove(outpath) # Remove zip file once extracted

    saved_name = '{}/{}'.format(s3def['sets']['dataset']['prefix'] , args.dataset)
    print('Save {} to {}/{}'.format(args.path, s3def['sets']['dataset']['bucket'], saved_name))
    if s3.PutDir(s3def['sets']['dataset']['bucket'], args.path, saved_name):
        shutil.rmtree(args.path, ignore_errors=True)

    url = s3.GetUrl(s3def['sets']['dataset']['bucket'], saved_name)
    print("Complete. Results saved to {}".format(url))

if __name__ == '__main__':
  args, unparsed = parser.parse_known_args()
  
  if args.debug:
      print("Wait for debugger attach")
      import ptvsd
      # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
      # Launch applicaiton on remote computer: 
      # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
      # Allow other computers to attach to ptvsd at this IP address and port.
      ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
      # Pause the program until a remote debugger is attached

      ptvsd.wait_for_attach()

      print("Debugger attached")

  main(args)