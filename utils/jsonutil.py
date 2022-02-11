import os
import json
import yaml

def WriteDictJson(outdict, path):

    jsonStr = json.dumps(outdict, indent=2, sort_keys=False)
    with open(path,"w") as f:
        f.write(jsonStr)
    return True

def ReadDictJson(filepath):
    jsondict = None
    try:
        with open(filepath) as json_file:
            jsondict = json.load(json_file)
        if not jsondict:
            print('Failed to load {}'.format(filepath))
    except:
        print('Failed to load {}'.format(filepath))
    return jsondict

def Dict2Json(outdict):
    jsonStr = json.dumps(outdict, sort_keys=False, indent=4)      
    return jsonStr

def Json2Dict(json_file):
    jsondict = json.load(json_file)
    return jsondict

def ReadDictYaml(filepath):
    yamldict = {}
    try:
        with open(filepath) as yaml_file:
            yamldict = yaml.safe_load(yaml_file)
        if not yamldict:
            print('Failed to load {}'.format(filepath))
    except ValueError:
        print('Failed to load {} error {}'.format(filepath, ValueError))
    return yamldict

def WriteDictYaml(outdict, path):
    yamlStr = yaml.dump(outdict, indent=2, sort_keys=False)
    with open(path,"w") as f:
        f.write(yamlStr)
    return True

def ReadDict(filepath):
    ext = os.path.splitext(filepath)[1]
    if ext=='.yaml':
        readDict = ReadDictYaml(filepath)
    elif ext=='.json':
        readDict = ReadDictJson(filepath)
    else:
        readDict = None
    return readDict

def WriteDict(outdict, filepath):
    ext = os.path.splitext(filepath)[1]
    if ext=='.yaml':
        readDict = WriteDictYaml(outdict, filepath)
    elif ext=='.json':
        readDict = WriteDictJson(outdict, filepath)
    else:
        readDict = None
    return readDict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return not(v==0)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')