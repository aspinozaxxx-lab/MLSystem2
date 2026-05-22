package io.geoalert.mapflow.service

import scala.concurrent.Await
import scala.concurrent.ExecutionContext
import scala.concurrent.duration._

import akka.actor.ActorSystem
import akka.http.scaladsl.model.HttpEntity
import akka.stream.scaladsl.Sink
import doobie.implicits._

import io.geoalert.mapflow.AkkaSystem
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.model.AoiFilter
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.AoiUtil
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.UserUtil

import geotrellis.proj4.LatLng

class ResultServiceSpec extends DbIntegrationTest {
  implicit val ec: ExecutionContext = ExecutionContext.global

  implicit val system: ActorSystem = AkkaSystem.system

  describe("ResultService") {
    it("Should provide result for downloading by processing") {
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)
      AoiUtil.createAois(
        processing,
        GeometryUtil.fromExtent(83, 54, 83.01, 54.01).withSRID(LatLng.epsgCode.get),
      )(UserUtil.regularUser)

      val service = ResultService(new ExportGeojsonServiceMock())
      val result = service.downloadResultByProcessing(processing.id).transact(xa).unsafeRunSync()
      val sink = Sink.fold[String, HttpEntity.ChunkStreamPart]("")(_ + _.data().utf8String)
      val r = Await.result(result.runWith(sink), 60.second)
      r should be("{}")
    }

    it("Should provide result for downloading by aoi") {
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)
      val aois = AoiUtil.createAois(
        processing,
        GeometryUtil.fromExtent(83, 54, 83.01, 54.01).withSRID(LatLng.epsgCode.get),
      )(UserUtil.regularUser)

      val service = ResultService(new ExportGeojsonServiceMock())
      val result = service.downloadResultByAoi(aois.head.id).transact(xa).unsafeRunSync()
      val sink = Sink.fold[String, HttpEntity.ChunkStreamPart]("")(_ + _.data().utf8String)
      val r = Await.result(result.runWith(sink), 60.second)
      r should be("{}")
    }
  }
}
