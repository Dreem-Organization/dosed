from settings import MINIMUM_EXAMPLE_SETTINGS
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm


bucket_name = 'dreem-dosed-minimum-example'

client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

files = client.list_objects(Bucket='dreem-dosed-minimum-example')["Contents"]
print("\n Downloading EDF files and annotations from S3")
for file in tqdm.tqdm(files):
    filename = file["Key"]
    client.download_file(
        Bucket=bucket_name,
        Key=filename,
        Filename=MINIMUM_EXAMPLE_SETTINGS["download_directory"] + "/{}".format(filename)
    )
