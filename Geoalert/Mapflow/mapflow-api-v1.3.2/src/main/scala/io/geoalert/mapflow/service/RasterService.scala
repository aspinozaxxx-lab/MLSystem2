package io.geoalert.mapflow.service

import java.util.Collections

import scala.concurrent.ExecutionContext

import com.amazonaws.services.s3.AmazonS3URI
import io.minio.MinioClient
import io.minio.http.Method

import io.geoalert.mapflow.DefaultExternalSystemConfig
import io.geoalert.mapflow.MiscConfig

class RasterService extends DefaultExternalSystemConfig with MiscConfig {
  implicit val ec: ExecutionContext = ExecutionContext.global

  val s3 = new MinioClient(s3Url, s3AccessKey, s3SecretKey)

  def generatePresignedUrl(uri: String): String = {
    val s3Uri = new AmazonS3URI(uri)

    s3.getPresignedObjectUrl(
      Method.GET,
      s3Uri.getBucket,
      s3Uri.getKey,
      24 * 60 * 60,
      Collections.emptyMap(),
    )
  }
}

object RasterService {
  def apply(): RasterService = new RasterService()
}
