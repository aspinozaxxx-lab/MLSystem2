import os
import tempfile
from typing import List

import boto3
import minio
from fastapi import UploadFile
from loguru import logger
from minio.error import S3Error

from config import Config
from functional.urlparse import parse_image_url

MINIO_HOST = Config.MINIO_HOST
MINIO_PORT = Config.MINIO_PORT
MINIO_ACCESS_KEY = 
MINIO_SECRET_KEY = 


class Minio:

    def __init__(self, minio_host=MINIO_HOST, minio_port=MINIO_PORT,
                 access_key=, secret_key=, **kwargs):

        # Initialize minioClient with an endpoint and access/secret keys.
        MINIO_URI = '{host}:{port}' if minio_port else '{host}'
        self.client = minio.Minio(
            MINIO_URI.format(host=minio_host, port=minio_port),
            access_key=,
            secret_key=,
            secure=False,
            **kwargs,
        )

    def upload(self, bucket: str, filename: str, file: UploadFile, path: str):
        # file is uploaded as UploadFile
        # Make a bucket with the make_bucket API call.
        try:
            found = self.client.bucket_exists(bucket)
            if not found:
                self.client.make_bucket(bucket)
        except S3Error as e:
            raise e
        # s3 path: corresponding folder of user and mosaic inside user folder
        s3_path = f"{path}/{filename}"
        file_size = os.fstat(file.file.fileno()).st_size
        self.client.put_object(bucket, s3_path, file.file, file_size)

    def f_put(self, bucket: str, filename: str, s3_mosaic_path: str, disk_image_path: str):
        try:
            found = self.client.bucket_exists(bucket)
            if not found:
                self.client.make_bucket(bucket)
        except S3Error as e:
            raise e
        self.client.fput_object(bucket_name=bucket, object_name=f'{s3_mosaic_path}/{filename}',
                                file_path=disk_image_path)

    def download(self, bucket_name: str, object_name: str, file_path: str):
        logger.info(f"Downloading file {file_path} from bucket:{bucket_name} filename:{object_name}.")
        self.client.fget_object(bucket_name, object_name, file_path)
        logger.info(f"File {file_path} have been downloaded.")

    def remove_object(self, bucket_name: str, objects: List[str]):
        for obj in objects:
            self.client.remove_object(bucket_name=bucket_name, object_name=obj)

    # convenience functions to handle minio objects by full url
    def remove_objects_by_url(self, urls: List[str]):
        for url in urls:
            if url is None:
                continue
            _, bucket_name, object_name = parse_image_url(url)
            self.client.remove_object(bucket_name=bucket_name, object_name=object_name)

    def download_object_by_url(self, url, file_path):
        _, bucket_name, object_name = parse_image_url(url)
        self.download(bucket_name=bucket_name, object_name=object_name, file_path=file_path)

    def upload_object_by_url(self, url: str, local_path: str):
        _, bucket_name, object_name = parse_image_url(url)
        try:
            found = self.client.bucket_exists(bucket_name)
            if not found:
                self.client.make_bucket(bucket_name)
        except S3Error as e:
            raise e
        self.client.fput_object(bucket_name=bucket_name, object_name=object_name,
                                file_path=local_path)

    def check_if_object_exists(self, url: str):
        # stat_object(bucket_name, object_name)
        _, bucket_name, object_name = parse_image_url(url)
        try:
            self.client.stat_object(bucket_name=bucket_name, object_name=object_name)
            return True
        except S3Error:
            return False


def save_file_to_disk(file: UploadFile, filename: str):
    temp_file = f"{tempfile.gettempdir()}/{filename}"
    with open(temp_file, "wb+") as file_object:
        file_object.write(file.file.read())
    return temp_file


class Boto3:
    def __init__(self,
                 storage_host=MINIO_HOST,
                 storage_port=MINIO_PORT,
                 access_key=,
                 secret_key=
                 ):
        endpoint_url = f"http://{storage_host}:{storage_port}"
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=,
            aws_secret_access_key=
        )


minio_client = Minio()
MINIO_BUCKET = Config.MINIO_BUCKET
s3 = Boto3()
