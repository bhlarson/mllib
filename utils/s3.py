import os
import io
import glob
from pathlib import PurePath
from tqdm import tqdm
import natsort as ns
from minio import Minio
from minio.error import (ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists)


class s3store:

    def __init__(self, address, access_key, secret_key, secure = False):
        self.s3 = Minio(address,
             access_key=access_key,
             secret_key=secret_key,
             secure=secure)
        
    def ListBuckets(self):
        return self.s3.list_buckets()

    def PutDir(self, bucket, path, setname):
        success = True
        files = glob.glob(glob.escape(path)+'/**/*.*', recursive=True)
        try:
            self.s3.make_bucket(bucket)
        except BucketAlreadyOwnedByYou as err:
            pass
        except BucketAlreadyExists as err:
            pass
        except ResponseError as err:
            print(err)
            raise err

        try:
            for file in tqdm(files):
                filename = setname+'/'+file.lstrip(path)
                self.s3.fput_object(bucket, filename, file)
        except ResponseError as err:
            print(err)

        return success

    def GetDir(self, bucket, setname, destdir):
        success = True

        # List all object paths in bucket that begin with my-prefixname.
        try:
            objects = self.s3.list_objects(bucket, prefix=setname, recursive=True)
            fileCount = len(tuple(objects))
            if(fileCount > 0):
                objects = self.s3.list_objects(bucket, prefix=setname, recursive=True) # Recreate object to reset iteratorator to beginning
                for obj in tqdm(objects, total=fileCount):
                    try:
                        destination = '{}/{}'.format(destdir,obj.object_name.lstrip(setname))
                        if obj.is_dir:
                            if not os.path.isdir(destination):
                                os.mkdir(destination)
                        else:
                            self.s3.fget_object(bucket, obj.object_name, destination)
                    except:
                        print('Failed to copy {}/{} to {}'.format(bucket, obj.object_name, destination))
                        success = False

            #fileCount = len(tuple(objects)) 
            #if(fileCount > 0):
            #    for obj in tqdm(objects):                       
            #        self.s3.fget_object(bucket, obj.object_name, '{}/{}'.format(destdir,obj.object_name.lstrip(setname)))
            #else:
            #    print('{}/{} contains {} objects'.format(bucket, setname, fileCount))

        except ResponseError as err:
            print(err)
            raise err
        except:
            print('Failed to read {}/{}'.format(bucket, setname))
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
                        destination = '{}/{}'.format(destdir,obj.object_name.lstrip(setname+'/'))
                        if obj.is_dir:
                            if not os.path.isdir(destination):
                                os.mkdir(destination)
                        else:
                            if not os.path.isfile(destination):
                                print('Copy {}/{} to {}'.format(bucket, obj.object_name, destination))                
                                self.s3.fget_object(bucket, obj.object_name, destination)
                    except:
                        print('Failed to copy {}/{} to {}'.format(bucket, obj.object_name, destination))
                        success = False

        except ResponseError as err:
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
            for obj in objects:
                if PurePath(obj.object_name).match(pattern):
                    files.append(obj.object_name)
        except ResponseError as err:
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
            print(response)
            data = response.data
        except ResponseError as err:
                print(err)
                raise err
        finally:
            if response:
                response.close()
                response.release_conn()

        return data

    def PutObject(self, object_name, obj):
        success = True

        # List all object paths in bucket that begin with my-prefixname.
        try:
            objStream = io.BytesIO(obj)
            self.s3.put_object(bucket, object_name, objStream, file_size=len(objStream))
        except ResponseError as err:
            print(err)
            raise err

        return response

    def ListModels(self, bucket='savedmodel', model_type='lit' ):
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
                    mpp = 1e-10*float(splitname[fields-1].replace('/',''))
                    if(basemodel not in sized_models):
                        sized_models[basemodel] = []

                    model = {'name':obj.object_name,
                             'resolution':resolution,
                             'mpp':mpp,
                    }
                    sized_models[basemodel].append(model)



        except:
            print('Failed to read objects')
        return models, sized_models

    def ListModelsWithResults(self, bucket='savedmodel', model_type='lit', result_name='results.json'):
        models_results = []
        models = self.ListModels(bucket=bucket, model_type=model_type)
        try:
            for model in models:
                objects = self.s3.list_objects('savedmodel', prefix=model, recursive=False)
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
            self.s3.make_bucket(bucket)
        except BucketAlreadyOwnedByYou as err:
            pass
        except BucketAlreadyExists as err:
            pass
        except ResponseError as err:
            print(err)
            raise err

        try:
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
        except ResponseError as err:
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

        except ResponseError as err:
            print(err)
            raise err

        return checkpoints