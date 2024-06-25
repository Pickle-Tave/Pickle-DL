from jiheonkey import *
import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_S3_ACCESS_KEY,
    aws_secret_access_key=AWS_S3_SECRET_KEY,
    region_name='ap-northeast-2'
)

unique_user_id = "user123"
file_name = '20230813_172117.jpg'
bucket_name = 's3-jiheon'
s3_key = unique_user_id + '/' + file_name

# 파일 업로드
s3.upload_file(file_name, bucket_name, s3_key)

print(f"File {file_name} uploaded to bucket {bucket_name} with key {s3_key}")