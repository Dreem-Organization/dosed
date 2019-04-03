import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm

from settings import MINIMUM_EXAMPLE_SETTINGS

bucket_name = 'dreem-dosed-minimum-example'

client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

bucket_objects = client.list_objects(Bucket='dreem-dosed-minimum-example')["Contents"]
print("\n Downloading EDF files and annotations from S3")
for bucket_object in tqdm.tqdm(bucket_objects):
    filename = bucket_object["Key"]
    client.download_file(
        Bucket=bucket_name,
        Key=filename,
        Filename=MINIMUM_EXAMPLE_SETTINGS["download_directory"] + "/{}".format(filename)
    )
