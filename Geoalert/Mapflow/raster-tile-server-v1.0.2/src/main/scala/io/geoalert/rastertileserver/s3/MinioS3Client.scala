package io.geoalert.rastertileserver.s3

import com.typesafe.scalalogging.LazyLogging
import geotrellis.store.s3.{AmazonS3URI, S3ClientProducer}

import java.net.URI
import software.amazon.awssdk.services.s3.{S3Client, S3Configuration}
import io.geoalert.rastertileserver.Config._
import io.minio.messages.Item
import io.minio.{GetObjectArgs, ListObjectsArgs, MinioClient, Result}
import software.amazon.awssdk.auth.credentials.{AwsBasicCredentials, StaticCredentialsProvider}
import software.amazon.awssdk.http.apache.ApacheHttpClient
import software.amazon.awssdk.regions.Region

import java.lang
import java.nio.charset.StandardCharsets

object MinioS3Client extends LazyLogging {
  lazy val s3Client: S3Client = {
    val credentials = AwsBasicCredentials.create(minioAccessKey, minioSecretKey)

    val configuration = S3Configuration.builder()
      .pathStyleAccessEnabled(true)
      .build()

    S3Client.builder()
      .endpointOverride(new URI(minioEndpoint))
      .region(Region.US_EAST_1)
      .httpClientBuilder(ApacheHttpClient.builder())
      .serviceConfiguration(configuration)
      .credentialsProvider(StaticCredentialsProvider.create(credentials))
      .build()
  }

  private lazy val minioClient:MinioClient = MinioClient.builder()
    .endpoint(minioEndpoint)
    .credentials(minioAccessKey, minioSecretKey)
    .build()

  def listObjects(uri: String): lang.Iterable[Result[Item]] = {
    val s3Uri = new AmazonS3URI(uri)
    val listObjectsArgs = ListObjectsArgs.builder()
      .bucket(s3Uri.getBucket)
      .prefix(s3Uri.getKey)
      .recursive(true)
      .fetchOwner(false)
      .build()
    MinioS3Client.minioClient.listObjects(listObjectsArgs)
  }

  def getObject(uri: String): String = {
    val s3Uri = new AmazonS3URI(uri)
    val getObjectsArgs = GetObjectArgs.builder()
      .bucket(s3Uri.getBucket)
      .`object`(s3Uri.getKey)
      .build()

    val bytes = MinioS3Client.minioClient.getObject(getObjectsArgs).readAllBytes()
    new String(bytes, StandardCharsets.UTF_8)
  }

  def apply(): Unit = {
    import scala.jdk.CollectionConverters._

    S3ClientProducer.set(() => MinioS3Client.s3Client)

    val buckets:List[String] = MinioS3Client.s3Client.listBuckets()
      .buckets()
      .asScala
      .map(_.name())
      .toList

    logger.info(s"Initialized Minio. Available buckets are ${buckets.mkString(", ")}")
  }
}
