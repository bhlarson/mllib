import glob
import os
from minio import Minio
from minio.error import (ResponseError, BucketAlreadyOwnedByYou,
                         BucketAlreadyExists)

from tqdm import tqdm

class s3store:

    def __init__(self, address, access_key, secret_key, secure = False):
        self.s3 = Minio(address,
             access_key=access_key,
             secret_key=secret_key,
             secure=secure)

    def PutDir(self, bucket, path, setname):
        success = True
        files = glob.glob(glob.escape(path)+'/*.*')
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
            for file in files:
                filename = setname+'/'+os.path.basename(file)
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
                objects = self.s3.list_objects(bucket, prefix=setname, recursive=True)
                i = 0
                for obj in tqdm(objects):                       
                    self.s3.fget_object(bucket, obj.object_name, '{}/{}'.format(destdir,os.path.basename(obj.object_name)))
                    if fileCount < 200 or i%int(fileCount/100.0):
                        pb.print_progress_bar(i)
                    i+=1
                pb.print_progress_bar(i)
            else:
                print('{}/{} contains {} objects'.format(bucket, setname, fileCount))

        except ResponseError as err:
            print(err)
            raise err

        return success
