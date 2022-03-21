import os
import io
import glob
import json, yaml
from datetime import datetime, timedelta
from pathlib import Path, PurePath
from tqdm import tqdm
import natsort as ns
from minio import Minio
import urllib3
import certifi

from .jsonutil import ReadDict

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

class s3store:

    def __init__(self, address, access_key, secret_key, tls = True, cert_verify=True, cert_path = None, timeout=5.0):
        self.addresss = address
        self.tls = tls
        urllib3.disable_warnings()
        if(cert_verify):
            if(cert_path):
                cert_location = cert_path
            else:
                cert_location = certifi.where()
            http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=cert_location, timeout=timeout, maxsize=10, block=True)
        else:
            http = urllib3.PoolManager(cert_reqs='CERT_NONE', timeout=timeout, maxsize=10, block=True)

        self.s3 = Minio(address,
             access_key=access_key,
             secret_key=secret_key,
             secure=tls,
             http_client = http)

    def GetUrl(self, bucket, path,  expires=timedelta(hours=2)):

        url = self.s3.presigned_get_object(bucket, path, expires=expires)
        return url

    def ListBuckets(self):
        return self.s3.list_buckets()

    def MakeBucket(self, bucket, location='us-east-1', object_lock=False):
        if not self.s3.bucket_exists(bucket):
            self.s3.make_bucket(bucket, location, object_lock)

    def PutFile(self, bucket, file, setname):

        success = True
        results = None

        try:
            self.MakeBucket(bucket)
            filename = setname+'/'+ os.path.basename(file)
            results = self.s3.fput_object(bucket, filename, file)
        except Exception as err:
            print(err)
            success = False

        return success

    def PutDir(self, bucket, path, setname):
        success = True
        files = list(Path(path).rglob('**/*.*'))

        try:
            self.MakeBucket(bucket)
            if path[len(path)-1] != '/':
                path = path +'/'
            for file in tqdm(files, total=len(files)):
                # Create object directory structure
                object_name = setname+ remove_prefix(str(file), str(Path(path)))
                self.s3.fput_object(bucket, object_name.lstrip('/'), str(file))
        except Exception as err:
            print(err)
            success = False

        return success

    def GetDir(self, bucket, setname, destdir):
        fileCount = 0

        # List all object paths in bucket that begin with my-prefixname.
        try:
            objects = self.s3.list_objects(bucket, prefix=setname, recursive=True)
            fileCount = len(tuple(objects))
            if(fileCount > 0):
                objects = self.s3.list_objects(bucket, prefix=setname, recursive=True) # Recreate object to reset iteratorator to beginning
                for obj in tqdm(objects, total=fileCount):
                    try:
                        
                        destination = '{}/{}'.format(destdir,remove_prefix(obj.object_name, setname))
                        if obj.is_dir:
                            if not os.path.isdir(destination):
                                os.mkdir(destination)
                        else:
                            self.s3.fget_object(bucket, obj.object_name, destination)
                    except:
                        print('Failed to copy {}/{} to {}'.format(bucket, obj.object_name, destination))


        except Exception as err:
            print(err)
            fileCount = 0
        except:
            print('Failed to read {}/{}'.format(bucket, setname))
            fileCount = 0

        return fileCount

    def GetFile(self, bucket, object_name, destination):
        success = True

        try:                     
            self.s3.fget_object(bucket, object_name, destination)
        except Exception as err:
            print(err)
            success = False
        except:
            print('Failed to copy {}/{} to {}'.format(bucket, object_name, destination))
            success = False

        return success

    def Mirror(self, bucket, setname, destdir):
        success = True

        try:
            objects = self.s3.list_objects(bucket, prefix=setname, recursive=True)
            fileCount = len(tuple(objects)) # Determine length for feedback during download
            if(fileCount > 0):
                objects = self.s3.list_objects(bucket, prefix=setname, recursive=True) # Recreate object to reset iteratorator to beginning
                for obj in tqdm(objects, total=fileCount):
                    try:
                        objstr = remove_prefix(obj.object_name, setname)
                        destination = '{}/{}'.format(destdir,objstr)
                        if obj.is_dir:
                            if not os.path.isdir(destination):
                                os.mkdir(destination)
                        else:
                            if not os.path.isfile(destination):
                                self.s3.fget_object(bucket, obj.object_name, destination)
                    except:
                        print('Failed to copy {}/{} to {}'.format(bucket, obj.object_name, destination))
                        success = False

        except Exception as err:
            print(err)
            raise err
        except:
            print('Failed to read {}/{}'.format(bucket, setname))
            success = False

        return success

    def ListObjects(self, bucket, setname=None, pattern='**', recursive=False):
        success = True
        files = []
        # List all object paths in bucket that begin with my-prefixname.
        try:
            objects = self.s3.list_objects(bucket, prefix=setname, recursive=recursive)
            if pattern is not None and len(pattern) > 0:
                for obj in objects:
                    if PurePath(obj.object_name).match(pattern):
                        files.append(obj.object_name)
            else:
                for obj in objects:
                    files.append(obj.object_name)
        except Exception as err:
            print(err)
            raise err

        return files

    def GetObjects(self, bucket, objectNames):
        objects = []
        for obj in objectNames:
            try:
                objects.append(GetObject(self, bucket, obj))
            except:
                print('error reading {}/{}'.format(bucket, obj))
        return objects

    def GetObject(self, bucket, object_name):
        success = True
        response = None
        data = None
        # List all object paths in bucket that begin with my-prefixname.
        try:
            response = self.s3.get_object(bucket, object_name)
            data = response.data
        except Exception as err:
                print(err)
        except:
                print('error reading {}/{}'.format(bucket, object_name))
        finally:
            if response:
                response.close()
                response.release_conn()

        return data

    def PutObject(self, bucket, object_name, obj):
        success = True

        # List all object paths in bucket that begin with my-prefixname.
        try:
            self.MakeBucket(bucket)
            if not isinstance(obj, io.BytesIO):
                objStream = io.BytesIO(obj)
                self.s3.put_object(bucket, object_name, objStream, length=len(objStream.getvalue()))
            else:
                obj.seek(0)
                self.s3.put_object(bucket, object_name, data=obj, length=len(obj.getvalue()))

        except Exception as err:
            print(err)
            success = False
        except:
                success = False

        return success

    def GetDict(self, bucket, object_name):
        success = True
        response = None
        data_dict = None
        # List all object paths in bucket that begin with my-prefixname.
        try:
            response = self.s3.get_object(bucket, object_name)

            if response.data:
                ext = os.path.splitext(object_name)[1]
                if ext=='.yaml':
                    data_dict = yaml.safe_load(response.data)
                elif ext=='.json':
                    data_dict = json.loads(response.data)



        except Exception as err:
            print('s3 Response error {}'.format(err))
        except json.JSONDecodeError as err:
            print('JSONDecodeError {}'.format(err))
        except:
            print('error reading {}/{}'.format(bucket, object_name))
        finally:
            if response:
                response.close()
                response.release_conn()

        return data_dict

    def PutDict(self, bucket, object_name, dict_data):
        success = True

        # List all object paths in bucket that begin with my-prefixname.
        try:
            self.MakeBucket(bucket)
            obj = json.dumps(dict_data, sort_keys=False, indent=4).encode()

            objStream = io.BytesIO(obj)
            self.s3.put_object(bucket, object_name, objStream, length=len(obj))
        except Exception as err:
            print(err)
            success = False
        except:
                success = False
        return success

    def ListModels(self, bucket='', model_type='' ):
        models = []
        sized_models = {}
        try:
            objects = self.s3.list_objects(bucket, prefix='', recursive=False)
            for obj in objects:
                splitname = obj.object_name.split('-')
                fields = len(splitname)
                if splitname[fields-1] == model_type+'/' or splitname[fields-2] == model_type : # remove saved models with specialized size
                    modelResults = self.s3.list_objects(bucket, prefix=splitname, recursive=False)       
                    models.append(obj.object_name)
                    if obj.object_name not in sized_models:
                        sized_models[obj.object_name] = []
                elif splitname[fields-3] == model_type:
                    targetstring = splitname[fields-2]+'-'
                    basemodel = obj.object_name.replace(splitname[fields-2]+'-','')
                    resolution = splitname[fields-2].split('x')
                    resolution = [int(resolution[0]),int(resolution[1])]
                    if(basemodel not in sized_models):
                        sized_models[basemodel] = []

                    model = {'name':obj.object_name,
                             'resolution':resolution
                    }
                    sized_models[basemodel].append(model)



        except:
            print('Failed to read objects')
        return models, sized_models

    def ListModelsWithResults(self, bucket, model_type='', result_name='results.json'):
        models_results = []
        models = self.ListModels(bucket=bucket, model_type=model_type)
        try:
            for model in models:
                objects = self.s3.list_objects(bucket, prefix=model, recursive=False)
                for obj in objects:
                    if obj.object_name == model+result_name:
                        models_results.append(model)
                        break

        except:
            print('Failed to read objects')
        return models_results
    
    def ListCheckpoints(self, bucket='training'):
        checkpoints = []
        try:
            objects = self.s3.list_objects('training', prefix='', recursive=False)
            for obj in objects:                              
                checkpoints.append(obj.object_name)

        except:
            print('Failed to read objects')
        return checkpoints
    def PutCheckpoint(self, bucket, checkpoint_path, checkpoint_name):
        success = True

        try:
            self.MakeBucket(bucket)
            ckpt = tf.train.latest_checkpoint(checkpoint_path)
            ckptfiles = glob.glob(ckpt+'.*')
            eventfiles = glob.glob(checkpoint_path+'/events.out.tfevents.*.*')
            eventfiles= ns.natsorted(eventfiles,alg=ns.PATH)

            # Add newest event file to lists
            ckptfiles.append(checkpoint_path+'/checkpoint')
            if len(eventfiles)>0:
                ckptfiles.append(eventfiles[-1])
            for ckfile in ckptfiles:
                obj = '{}/{}'.format(checkpoint_name,os.path.basename(ckfile))
                print('Write {} to {}/{}'.format(ckfile, bucket,obj))
                self.s3.fput_object(bucket, obj, ckfile)
        except Exception as err:
            print(err)
            raise err

        return success

    def GetCheckpoint(self, bucket, checkpoint_name, checkpoint_path):
        success = True
        # List all object paths in bucket that begin with my-prefixname.
        try:
            objects = self.s3.list_objects(bucket, prefix=checkpoint_name, 
                                            recursive=True)
            for obj in objects:
                print(obj.bucket_name, obj.object_name.encode('utf-8'), obj.last_modified,
                        obj.etag, obj.size, obj.content_type)

                        
                self.s3.fget_object(bucket, obj.object_name, '{}/{}'.format(checkpoint_path,os.path.basename(obj.object_name)))

        except Exception as err:
            print(err)
            raise err

        return checkpoints

    def CloneObjects(self, destbucket, destsetname, srcS3, srcbucket, srcsetname):
        success = True

        # List all object paths in bucket that begin with my-prefixname.
        try:
            objects = srcS3.list_objects(srcbucket, prefix=srcsetname, recursive=True)
            fileCount = len(tuple(objects)) 
            if(fileCount > 0):
                objects = srcS3.list_objects(srcbucket, prefix=setname, recursive=True)
                for obj in tqdm(objects, total=fileCount):
                    try:
                        destination = '{}/{}'.format(destsetname,remove_prefix(obj.object_name,srcsetname))

                        srcobj = srcS3.GetObject(srcbucket, obj.object_name)
                        self.PutObject(destbucket, destination, srcobj)

                        #if obj.is_dir:
                        #    if not os.path.isdir(destination):
                        #        os.mkdir(destination)
                        #else:
                        #    self.s3.fget_object(bucket, obj.object_name, destination)
                    except:
                        print('Failed to copy from {}/{} to {}/{}'.format(srcbucket,obj.object_name, destbucket, destination))
                        success = False            
            else:
                print('{}/{} contains {} objects'.format(srcbucket, setname, fileCount))

        except Exception as err:
            print(err)
            raise err

        return success

    def RemoveObjects(self, bucket, setname=None, pattern='**', recursive=False):
        success = True

        # List all object paths in bucket that begin with my-prefixname.
        # remove_objects minio API not working
        try:
            objects = self.s3.list_objects(bucket, prefix=setname, recursive=recursive)
            for obj in objects:
                self.s3.remove_object(bucket, obj.object_name)

        except Exception as err:
            print(err)
            success = False

        return success

def Connect(credentials_filename='creds.json', s3_name='store'):

    creds = ReadDict(credentials_filename)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(credentials_filename))
        return False

    s3_creds = next(filter(lambda d: d.get('name') == s3_name, creds['s3']), None)
    
    s3 = s3store(s3_creds['address'], 
                 s3_creds['access key'], 
                 s3_creds['secret key'],
                 tls=s3_creds['tls'],
                 cert_verify=s3_creds['cert verify'],
                 cert_path=s3_creds['cert path'],
                )
                
    buckets = s3.ListBuckets()
    if not (len(buckets) >= 0) :
        s3 = None
        print('Failed connect to {}'.format(s3_creds['address']))

    del s3_creds['access key']
    del s3_creds['secret key']
    del s3_creds['cert verify']
    del s3_creds['cert path']
    
    return s3, creds, s3_creds