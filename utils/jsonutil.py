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
    except ValueError:
        print('Failed to load {} error {}'.format(filepath, ValueError))
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

def ReadDict(filepath):
    ext = os.path.splitext(filepath)
    readDict = None
    if ext=='.yaml':
        readDict = ReadDictYaml(filepath)
    elif ext=='.json':
        readDict = ReadDictJson(filepath)

    return readDict