from minio import Minio
from minio.commonconfig import Tags
import os

# Initialize MinIO client
client = Minio(
    "localhost:3900",
    access_key=os.getenv('BUCKET_ACCESS_KEY'),
    secret_key=os.getenv('BUCKET_SECRET_KEY'),
    secure=False,  # Set to True if using HTTPS
    region=os.getenv('BUCKET_REGION')
)
 
# Example of checking if the client can access a bucket
# uploadBucket = client.make_bucket("uploads")
buckets = client.list_buckets()

print("Buckets:", buckets)

# for bucket in buckets:
#     print(bucket.name)

def getPresignedUrls(object_name, bucket_name = 'temp'):
    url = client.presigned_put_object(bucket_name, object_name)
    return url

def getPresignedGetUrls(object_name, bucket_name = 'temp'):
    url = client.presigned_get_object(bucket_name, object_name)
    return url

def getAllFiles(bucket_name = 'temp'):
    files = client.list_objects(bucket_name)
    return files

def setMetaTags(object_name, tags, bucket_name = 'temp'):
    client.set_object_tags(bucket_name, object_name, tags)
