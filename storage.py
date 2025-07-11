import boto3
class S3Storage:
    def __init__(self, access, secret, bucket, region):
        self.bucket = bucket
        self.s3 = boto3.client(
            's3', aws_access_key_id=access, aws_secret_access_key=secret, region_name=region
        )

    def upload_file(self, binary: bytes, image_id: str):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f'images/{image_id}.jpg',
            Body=binary,
            ContentType='image/jpeg',
            ACL='private'
        )

    def get_image_binary(self, image_id: str) -> bytes:
        r = self.s3.get_object(Bucket=self.bucket, Key=f'images/{image_id}.jpg')
        return r['Body'].read()

    def presigned_url(self, image_id: str, expires=3600):
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': f'images/{image_id}.jpg'},
            ExpiresIn=expires
        )