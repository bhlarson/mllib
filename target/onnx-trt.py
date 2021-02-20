# tf2-onnx.py -> onnx-trt.py -> trt-tst.py

BATCH_SIZE = 32
import argparse
import os
import sys
import json
import shutil
import tensorrt as trt

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true',help='Wait for debuger attach')
parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
parser.add_argument('-savedmodelname', type=str, default="2021-02-19-20-51-59-cocoseg", help='Saved model name')
parser.add_argument('-targetname', type=str, default="model", help='Final model wiout extension')
parser.add_argument('-workdir', type=str, default="trt", help='Working directory')
parser.add_argument('-onnxname', type=str, default="model.onnx", help='Working directory')
parser.add_argument('-workspace_memory', type=int, default=4096, help='trtexec workspace size in megabytes')
parser.add_argument('-fp16', action='store_true', help='If set, Generate FP16 model.')


def main(args):
    failed = False

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
                 cert_verify=s3def['cert_verify'], 
                 cert_path=s3def['cert_path']
                 )

    workdir = '{}/{}'.format(args.workdir, args.savedmodelname)
    inobj = '{}/{}/{}'.format(s3def['sets']['model']['prefix'],args.savedmodelname, args.onnxname)
    objpath = '{}/{}'.format(s3def['sets']['model']['prefix'],args.savedmodelname)


    infile = '{}/{}'.format(workdir, args.onnxname)
    targetname = args.targetname
    if args.fp16:
        targetname += '-fp16'
    outfile = '{}/{}.trt'.format(workdir, targetname)
    logfile = '{}/{}-trt.log'.format(workdir, targetname)

    if not s3.GetFile(s3def['sets']['model']['bucket'], inobj, infile):
        print('Failed to load {}/{} to {}'.format(s3def['sets']['model']['bucket'], inobj, infile ))
        failed = True
        return failed

    params = ''
    if args.fp16:
        params = '--fp16'
    
    USE_FP16 = True

    # May need to shut down all kernels and restart before this - otherwise you might get cuDNN initialization errors:
    #if USE_FP16:
    #    os.system("trtexec --onnx=resnet50_onnx_model.onnx --saveEngine=resnet_engine.trt  --explicitBatch --fp16")
    #else:
    #    os.system("trtexec --onnx=resnet50_onnx_model.onnx --saveEngine=resnet_engine.trt  --explicitBatch")

    trtcmd = "trtexec --onnx={} --saveEngine={}  --explicitBatch --workspace={} {} --verbose  2>&1 | tee {}".format(
        infile, outfile, args.workspace_memory, params, logfile)

    failed = os.system(trtcmd)

    if not failed:
        if s3.PutFile(s3def['sets']['model']['bucket'], outfile, objpath):
            if s3.PutFile(s3def['sets']['model']['bucket'], logfile, objpath):
                shutil.rmtree(args.workdir, ignore_errors=True) 

    # trtcmd = "trtexec --onnx=/store/dmp/cl/store/mllib/model/2021-02-19-20-51-59-cocoseg/model.onnx --saveEngine=/store/dmp/cl/store/mllib/model/2021-02-19-20-51-59-cocoseg/model.trt  --explicitBatch --workspace=4096 --verbose  2>&1 | tee trtexe.log"
    print('onnx-trt complete return {}'.format(failed))
    return failed

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

      print("Debugger attach")

  main(args)