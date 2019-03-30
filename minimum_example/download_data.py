import os
from settings import MINIMUM_EXAMPLE_SETTINGS
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm


def download_dir(client, resource, local='/tmp', bucket='dreem-dosed-minimum-example'):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        for file in tqdm.tqdm(result.get('Contents', [])):
            dest_pathname = os.path.join(local, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)


client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
resource = boto3.resource('s3')
download_dir(client, resource, MINIMUM_EXAMPLE_SETTINGS["download_directory"])
