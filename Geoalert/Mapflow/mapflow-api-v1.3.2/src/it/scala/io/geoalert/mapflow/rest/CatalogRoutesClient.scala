package io.geoalert.mapflow.rest

import scala.concurrent.duration._

import akka.http.scaladsl.model.HttpHeader
import akka.http.scaladsl.testkit.RouteTestTimeout
import akka.http.scaladsl.testkit.ScalatestRouteTest
import akka.testkit.TestDuration

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.rest.json.Decoders
import io.geoalert.mapflow.rest.json.Encoders
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil

class CatalogRoutesClient
    extends DbIntegrationTest
       with ScalatestRouteTest
       with Encoders
       with Decoders {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(30.seconds.dilated)

  lazy val auth: HttpHeader =
    TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)
//  describe("Catalog API") {
//    it("should return a single image from each catalog client") {
//      val entity = ImageCatalogRequestJson(
//        aoi=Extent(0, 0, 10, 10).toPolygon(),
//        None,
//        None,
//        None,
//        None,
//        None,
//        None,
//        None,
//        None,
//      )
//
//      Post(s"/rest/catalog/meta", entity) ~> addHeader(
//        auth
//      ) ~!> HttpServer.route ~> check {
//        status should be(StatusCodes.OK)
//
//        val response = responseAs[ImageCatalogResponseJson]
//        response.images.size should be(1)
//        response.images.head.colorBandOrder should be("RGB")
//      }
//    }
//  }

}
