import boto3
import tqdm
from tqdm.notebook import tqdm
import ipywidgets

session = boto3.session.Session()
s3_client = session.client(
    service_name='s3',
    endpoint_url='https://hb.bizmrg.com',
    aws_access_key_id='puEM2LPKPZGpnLPMpqBDoM',
    aws_secret_access_key='cHDf7RXTFG82gwyXsQDh5zzVsMavbGBhVsbDL2V74DXs'
)

bucket_name = 'ds-aurora'
dir_d = 'datasets/DS-ICE.1.4/ice_water_v3/Data'
paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=dir_d)

keys = []
for page in pages:
    for key in page['Contents']:
        keys.append(key['Key'])
        # print(key['Key'])

for file_path in tqdm(keys):
    # file_name = os.path.basename(file_path)
    file_name = file_path.split('/')[-1]

    response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
    with open(f'E:/dafuck/data/{file_name}', 'wb') as f:
        f.write(response['Body'].read())
