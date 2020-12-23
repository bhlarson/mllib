import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('path', help='coco dataset path')
parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-dataset', type=str, default='coco', help='Dataset.')


cocourl=["http://images.cocodataset.org/zips/train2017.zip",
                "http://images.cocodataset.org/zips/val2017.zip",
                "http://images.cocodataset.org/zips/test2017.zip",
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/image_info_test2017.zip",
                "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
]
def main(args):

    creds = {}
    with open(args.credentails) as json_file:
        creds = json.load(json_file)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))

    s3def = creds['s3'][0]
    s3 = s3store(s3def['address'], s3def['access key'], s3def['secret key'])

    for url in cocourl:
        outpath = '{}/{}'.format(args.path,os.path.basename(url))
        if os.path.isfile(outpath):
            print('{} exists.  Skipping'.format(outpath))
        else:
            sysmsg = 'wget -O {} {} '.format(outpath, url)
            print(sysmsg)
            os.system(sysmsg)

        sysmsg = 'unzip {} -d {}'.format(outpath, args.path)
        print(sysmsg)
        os.system(sysmsg)

    saved_name = '{}/{}'.format(s3def['sets']['dataset']['prefix'] , args.dataset)
    print('Save model to {}/{}'.format(s3def['sets']['dataset']['bucket'],saved_name))
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