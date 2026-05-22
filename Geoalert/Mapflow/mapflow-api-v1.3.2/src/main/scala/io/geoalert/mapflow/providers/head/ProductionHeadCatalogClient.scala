package io.geoalert.mapflow.providers.head

import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.Date
import java.util.UUID
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.Failure

import _root_.io.geoalert.mapflow.rest.json.ImageCatalogRequestJson
import _root_.io.geoalert.mapflow.rest.json.ImageJson
import akka.http.scaladsl.client.RequestBuilding._
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.headers.RawHeader
import akka.stream.scaladsl.Sink
import akka.stream.scaladsl.Source
import cats.effect.ContextShift
import cats.effect.IO
import cats.implicits._
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto._

import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.exception.ExternalSystemError
import io.geoalert.mapflow.util.HttpUtils

import geotrellis.proj4.CRS
import geotrellis.vector._

case class LoginRequest(login: String, pwd: String)
case class LoginResponse(
    code: Int,
    msg: String,
    data: Option[LoginPayload],
  )
case class LoginPayload(ticket: String)

case class HeadCoordinates(lat: Double, lng: Double)
case class HeadGeometry(cors: Seq[HeadCoordinates], `type`: Int = 4)

case class HeadSensor(`type`: Int = 1)
case class HeadCondition(
    fdate: String,
    tdate: String,
    cc: Double,
    ac: Double,
    ona: Double,
    sza: Double,
  )
case class HeadAoi(name: String, geometry: HeadGeometry)

object HeadAoi {
  def apply(aoi: Geometry): HeadAoi =
    aoi.geom match {
      case poly: Polygon =>
        val coords =
          poly.getExteriorRing.getCoordinates.map(crd => HeadCoordinates(crd.y, crd.x)).toSeq
        HeadAoi(UUID.randomUUID().toString, HeadGeometry(coords))
      case _ =>
        throw new BadRequest("Head AOI must be a Polygon")
    }
}

case class HeadPage(start: Int = 0, length: Int = 1000)
case class SearchRequest(
    aois: Seq[HeadAoi],
    condition: HeadCondition,
    page: HeadPage = HeadPage(),
    sensor: HeadSensor = HeadSensor(),
    satIds: Seq[Int] = Seq(60, 51, 52, 53, 54, 55, 56, 58, 61, 62, 63, 64, 65, 68, 70, 72, 74, 78,
      76, 57, 12, 19, 20, 21, 38, 39, 40, 41, 42, 43, 44, 45, 77, 59, 46, 47, 48, 66, 67, 69, 71,
      73, 75, 50, 18, 17, 29, 30, 31, 32, 33, 34, 28, 13, 14, 15, 16),
  )

case class SearchResponse(
    code: Int,
    msg: String,
    data: Option[SearchPayload],
  )

case class SearchPayload(images: Seq[HeadImage])

case class HeadImage(
    sceneId: String,
    boundaries: Seq[(Double, Double)],
    cc: Double,
    ac: Double,
    ona: Double,
    sza: Double,
    resolution: Double,
    satName: String,
    time: String,
    productQuality: String,
  ) {
  val previewUrl: String = {
    val extent =
      Polygon(boundaries).extent.reproject(CRS.fromEpsgCode(4326), CRS.fromEpsgCode(3857))
    val width = ((extent.xmax - extent.xmin) / 100).floor.toInt
    val height = ((extent.ymax - extent.ymin) / 100).floor.toInt
    val format = URLEncoder.encode("image/png", "utf-8")
    val layers = URLEncoder.encode(s"rs_data:${sceneId}", "utf-8")
    val crs = URLEncoder.encode("EPSG:3857", "utf-8")
    val bbox =
      URLEncoder.encode(s"${extent.xmin},${extent.ymin},${extent.xmax},${extent.ymax}", "utf-8")
    s"https://home.sat-imagery.com/geoserver/rs_data/wms?SERVICE=WMS&VERSION=1.1.0&REQUEST=GetMap&FORMAT=${format}&TRANSPARENT=true&LAYERS=${layers}&STYLES=&serverType=geoserver&crossOrigin=anonymous&tiled=false&angle=0&WIDTH=${width}&HEIGHT=${height}&SRS=${crs}&BBOX=${bbox}"
  }
}
class ProductionHeadCatalogClient(headProviderName: String)
    extends HeadCatalogClient
       with LazyLogging {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  val baseUrl = "https://home.sat-imagery.com"

  private val connectionFlow = HttpUtils.httpConnection(new URL(baseUrl))
  val loginUrl = "/rssc-portal/satimagery/account/login"
  val searchUrl = "/rssc-portal/satimagery/archive/search?sort=time,desc"

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        8,
        new ThreadFactoryBuilder().setNameFormat("head-catalog-client-%d").build(),
      )
    )
  )

  private def authenticate(username: String, password:  IO[String] = {
    val request = Post(loginUrl, LoginRequest(username, password))

    val future = for {
      response <- runRequest(request)
      payload <- HttpUtils.parseResponse[LoginResponse](response, "HEAD")
    } yield
      if (payload.data.isEmpty)
        throw ExternalSystemError("Cannot login to HEAD provider")
      else
        payload.data.get.ticket

    IO.fromFuture(IO(future))
  }

  val headDateFormat = new SimpleDateFormat("yyyy-MM-dd")

  val headDatetimeFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")

  def searchMeta(
      username: String,
      password: ,
      input: ImageCatalogRequestJson,
    ): IO[List[ImageJson]] = {
    def dateToString(date: Instant): String =
      headDateFormat.format(Date.from(date))

    def stringToInstant(date: String): Instant =
      headDatetimeFormat.parse(date).toInstant

    def boundariesToPolygon(points: Seq[(Double, Double)]): Geometry =
      Polygon(points)

    def convert(head: HeadImage): ImageJson =
      ImageJson(
        head.sceneId,
        boundariesToPolygon(head.boundaries),
        head.resolution,
        stringToInstant(head.time),
        head.productQuality,
        head.satName,
        "unknown",
        head.cc / 100,
        head.ona,
        Some("png"),
        Some(head.previewUrl),
        Some(headProviderName),
      )

    def headAois(aoi: Geometry): Seq[HeadAoi] =
      aoi match {
        case poly: Polygon =>
          Seq(HeadAoi(poly))
        case mpoly: MultiPolygon =>
          mpoly.polygons.map(HeadAoi(_))
        case _ =>
          throw new BadRequest("Polygon or MultiPolygon AOI is expected")
      }

    def search(token: , input: ImageCatalogRequestJson): IO[List[ImageJson]] = {
      def searchPolygon(aoi: Polygon): IO[List[ImageJson]] = {
        val body = SearchRequest(
          headAois(aoi),
          HeadCondition(
            dateToString(
              input.acquisitionDateFrom.getOrElse(Instant.now().minus(365 * 10, ChronoUnit.DAYS))
            ),
            dateToString(input.acquisitionDateTo.getOrElse(Instant.now())),
            input.maxCloudCover.getOrElse(100),
            input.minAoiIntersectionPercent.getOrElse(0),
            input.minOffNadirAngle.getOrElse(90),
            0,
          ),
        )
        val request = Post(searchUrl, body).withHeaders(Seq(RawHeader("token", token)))

        val future = for {
          response <- runRequest(request)
          payload <- HttpUtils.parseResponse[SearchResponse](response, "HEAD")
          images = payload.data.map(_.images).getOrElse(Seq())
        } yield images.map(convert).toList
        logger.debug(s"Sending request for AOI ${aoi.toString}")
        IO.fromFuture(IO(future))
      }

      input.aoi match {
        case poly: Polygon => searchPolygon(poly)
        case multiPoly: MultiPolygon =>
          multiPoly
            .polygons
            .toList
            .traverse[IO, List[ImageJson]](searchPolygon)
            .map(_.flatten)
            .map(_.distinct)
        case _ => IO.pure(List[ImageJson]())
      }
    }

    for {
      token <- authenticate(username, password)
      response <- search(token, input)
    } yield response
  }

  private def runRequest(request: HttpRequest): Future[HttpResponse] = {
    val requestFuture = Source
      .single(request)
      .via(connectionFlow)
      .runWith(Sink.head)

    requestFuture.onComplete {
      case Failure(e) => logger.error(s"Error trying to call WE", e)
      case _ =>
    }

    Future.firstCompletedOf(
      Seq(
        akka
          .pattern
          .after(3.minutes, system.scheduler)(
            Future.failed(new RuntimeException("WorkflowEngine request timeout"))
          ),
        requestFuture,
      )
    )
  }
}
