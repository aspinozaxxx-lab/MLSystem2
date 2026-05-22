package io.geoalert.rastertileserver

import com.typesafe.scalalogging.LazyLogging
import io.geoalert.rastertileserver.s3.MinioS3Client


object Application extends App with LazyLogging {
  MinioS3Client()
  HttpServer()
}
