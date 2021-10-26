import os
import sys
import shutil
import subprocess
import copy
import io
import json
import tempfile
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, os.path.abspath(''))
from utils.jsonutil import WriteDictJson, ReadDictJson

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    
    parser.add_argument('-out', type=str, default='mllib', help='Output path')
    parser.add_argument('-config', type=str, default='build.json', help='Configuration file')
    parser.add_argument('-version', type=str, default='latest', help='Version number')

    parser.add_argument('-archive', type=str, default='localhost:32000', help='destination archive')
    parser.add_argument('-dirs', type=json.loads, default=None, help='List of directories to create')
    parser.add_argument('-collect', type=json.loads, default=None, help='List of copies to perform')
    parser.add_argument('-zip', type=json.loads, default=None, help='files to add to zip')
    parser.add_argument('-whl', type=json.loads, default=None, help='List of whl to package')
    parser.add_argument('-dockerbuild', type=json.loads, default=None, help='List dockerfiles to build')
    parser.add_argument('-dockerpull', type=json.loads, default=None, help='List cocker images to pull')
    

    args = parser.parse_args()

    if type(args.config) ==str:
        config = ReadDictJson(args.config)
        if 'dirs' in config:
            if args.dirs is None:
                args.dirs = []
            args.dirs.extend(config['dirs'])
        if 'collect' in config:
            if args.collect is None:
                args.collect = []
            args.collect.extend(config['collect'])
        if 'whl' in config:
            if args.whl is None:
                args.whl = []
            args.whl.extend(config['whl'])

        if 'dockerbuild' in config:
            if args.dockerbuild is None:
                args.dockerbuild = []
            args.dockerbuild.extend(config['dockerbuild'])


    return args

def cmd(command):
    print('$ '+command)
    initial = datetime.now()
    result = subprocess.call(command, shell=True)
    dt = (datetime.now()-initial).total_seconds()
    print('Command complete {}s'.format(dt))
    return result

def Collect(copies, path, symlinks=True, dirs_exist_ok=True):
    if copies is not None and type(copies) is list:
        for copy_dfn in copies:
            ignore=None
            if "ignore" in copy_dfn:
                ignore=copy_dfn['ignore']
                
            '''shutil.copytree(
                copy_dfn['src'],
                '{}/{}'.format(path, copy_dfn['dest']), 
                ignore=ignore)'''
            copycmd = 'cp -a {} {}/{}'.format(copy_dfn['src'],path,copy_dfn['dest'])
            cmd(copycmd)

def GetWhl(whls, dest):
    if whls is not None and type(whls) is list:
        for whl in whls:
            pipcmd = 'pip3 download {}'.format(whl['name'])

            if 'version' in whl:
                pipcmd += '=={}'.format(whl['version'])
            if 'archive' in whl:
                pipcmd += " -i {}".format(whl['archive'])

            pipcmd += ' --dest {}'.format(dest)
            cmd(pipcmd)

def DockerBuild(dockers, archive, version):
    result = 0
    if dockers is not None and type(dockers) is list:
        for docker in dockers:
            docker_name = '{}/{}:{}'.format(archive,docker['name'],version)
            print('Docker build {}'.format(docker_name))
            dockercmd = 'docker build --pull -f {} -t {} "{}"'.format(
                docker['dockerfile'], docker_name, docker['context'])
            result=cmd(dockercmd)
            if result != 0:
                print('Failed {}={}.  Exiting.'.format(dockercmd, result))
                break

            dockercmd = 'docker push {}'.format(docker_name)
            result = cmd(dockercmd)
            if result != 0:
                print('Failed {}={}.  Exiting.'.format(dockercmd, result))
                break
            print('Image {} succeeded'.format(docker_name))
    return result

def main(args):
    initial = datetime.now()
    print("Build mlworkflow archive: {} version: {}".format(args.archive, args.version))

    result = DockerBuild(args.dockerbuild, args.archive, args.version)
    dt = (datetime.now()-initial).total_seconds()
    print("Build complete succeeded={} {}s".format(result==0, dt))

if __name__ == '__main__':
    import argparse
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy

        debugpy.listen(address=('0.0.0.0', args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    main(args)