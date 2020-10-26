import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('path', help='coco dataset path')


cocourl=["http://images.cocodataset.org/zips/train2017.zip",
                "http://images.cocodataset.org/zips/val2017.zip",
                "http://images.cocodataset.org/zips/test2017.zip",
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/image_info_test2017.zip",
                "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
]
def main(unparsed):
    for url in cocourl:
        outpath = '{}/{}'.format(FLAGS.path,os.path.basename(url))
        if os.path.isfile(outpath):
            print('{} exists.  Skipping'.format(outpath))
        else:
            sysmsg = 'wget -O {} {} '.format(outpath, url)
            print(sysmsg)
            os.system(sysmsg)

        sysmsg = 'unzip {} -d {}'.format(outpath, FLAGS.path)
        print(sysmsg)
        os.system(sysmsg)

    print("Complete. Results saved to {}".format(FLAGS.path))

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  
  if FLAGS.debug:
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

  main(unparsed)