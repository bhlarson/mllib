import os
import json
import yaml

def WriteDictJson(outdict, path):

    jsonStr = json.dumps(outdict, indent=2, sort_keys=False)
    f = open(path,"w")
    f.write(jsonStr)
    f.close()
       
    return True

def ReadDictJson(filepath):
    jsondict = {}
    try:
        with open(filepath) as json_file:
            jsondict = json.load(json_file)
        if not jsondict:
            print('Failed to load {}'.format(filepath))
    except:
        print('Failed to load {}'.format(filepath))
    return jsondict

def Dict2Json(dict):
    jsonStr = json.dumps(outdict, sort_keys=False, indent=4)      
    return jsonStr

def Json2Dict(json):
    jsondict = json.load(json_file)
    return jsondict


def ReadDictYaml(filepath):
    yamldict = {}
    try:
        with open(filepath) as yaml_file:
            yamldict = yaml.safe_load(yaml_file)
        if not yamldict:
            print('Failed to load {}'.format(filepath))
    except:
        print('Failed to load {}'.format(filepath))
    return yamldict

def ReadDict(filepath):
    ext = os.path.splitext(filepath)
    if ext=='yaml':
        readDict = ReadDictYaml(filepath)
    elif ext=='json':
        workflow = ReadDictJson(filepath)

    return readDict