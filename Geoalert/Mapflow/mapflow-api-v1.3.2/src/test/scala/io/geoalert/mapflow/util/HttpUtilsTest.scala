package io.geoalert.mapflow.util

import akka.http.scaladsl.model.{ContentTypes, HttpEntity, HttpResponse}
import io.geoalert.mapflow.service.we.model.WorkflowResponse
import org.scalatest.{FunSpec, Matchers}
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto._

import java.time.{LocalDateTime, ZoneOffset}
import scala.concurrent.{Await, ExecutionContext, ExecutionContextExecutor}
import scala.concurrent.duration._

class HttpUtilsTest extends FunSpec with Matchers {
  implicit val ec: ExecutionContextExecutor = ExecutionContext.global

  describe("HttpUtils") {
    it("should parse WE response") {
      val body = """
                   |  {
                   |    "id": 2400112,
                   |    "workflowDefinitionId": 2320447,
                   |    "stages": [
                   |      {
                   |        "id": 2400114,
                   |        "name": "select-source",
                   |        "description": "Source selection",
                   |        "status": "OK",
                   |        "statusUpdateDate": "2021-03-14T17:00:19.458"
                   |      },
                   |      {
                   |        "id": 2400116,
                   |        "name": "load-data",
                   |        "description": "Downloading data",
                   |        "status": "OK",
                   |        "taskIds": [
                   |          2400125
                   |        ],
                   |        "statusUpdateDate": "2021-03-14T17:00:35.456"
                   |      },
                   |      {
                   |        "id": 2400118,
                   |        "name": "inference",
                   |        "description": "Detecting buildings",
                   |        "status": "OK",
                   |        "taskIds": [
                   |          2400128
                   |        ],
                   |        "statusUpdateDate": "2021-03-14T17:02:52.42"
                   |      },
                   |      {
                   |        "id": 2400120,
                   |        "name": "import-vector",
                   |        "description": "Saving results",
                   |        "status": "OK",
                   |        "taskIds": [
                   |          2400160
                   |        ],
                   |        "statusUpdateDate": "2021-03-14T17:02:58.374"
                   |      }
                   |    ],
                   |    "areasOfInterest": [
                   |      {
                   |        "id": 2400113,
                   |        "geometry": {
                   |          "type": "Polygon",
                   |          "coordinates": [
                   |            [
                   |              [
                   |                47.34910345962362,
                   |                42.93416295605251
                   |              ],
                   |              [
                   |                47.3713839173918,
                   |                42.93450935622563
                   |              ],
                   |              [
                   |                47.37170155897356,
                   |                42.92355711800198
                   |              ],
                   |              [
                   |                47.34942110120539,
                   |                42.923210656229976
                   |              ],
                   |              [
                   |                47.34910345962362,
                   |                42.93416295605251
                   |              ]
                   |            ]
                   |          ]
                   |        }
                   |      }
                   |    ],
                   |    "rasterLayer": null,
                   |    "vectorLayer": {
                   |      "id": 2400159,
                   |      "uri": "https://vector-staging.mapflow.ai/api/layers/64be4ec6-471c-4439-87cf-5b6c8401f5d6.json",
                   |      "layerId": "64be4ec6-471c-4439-87cf-5b6c8401f5d6"
                   |    },
                   |    "artifacts": [
                   |      {
                   |        "areaOfInterestId": 2400113,
                   |        "artifactType": "RAW_RASTER",
                   |        "uri": "s3://workflow-white-maps/workflow-2400112/b626e548-188c-40c4-aa20-06f0a8833d41/area-2400113.tif"
                   |      },
                   |      {
                   |        "areaOfInterestId": 2400113,
                   |        "artifactType": "RAW_VECTOR",
                   |        "uri": "s3://workflow-white-maps/workflow-2400112/4cbb31d6-7a08-4abe-bbba-134553846375/area-2400113.geojson"
                   |      }
                   |    ],
                   |    "status": "OK",
                   |    "statusUpdateDate": "2021-03-14T17:02:58.426",
                   |    "createDate": "2021-03-14T17:00:19.043",
                   |    "system": "white-maps",
                   |    "processingId": "3ae106af-2465-43d7-a497-a90bd0633fcb",
                   |    "params": {
                   |      "raster-layer-uri": "s3://white-maps-rasters/4b3fa1ba-e806-4413-b19a-e58e329f0fc7",
                   |      "priority": "9",
                   |      "vector-layer-id": "64be4ec6-471c-4439-87cf-5b6c8401f5d6"
                   |    },
                   |    "meta": {}
                   |  }
                   |""".stripMargin


      val response = HttpResponse(entity = HttpEntity(body).withContentType(ContentTypes.`application/json`))
      val f = HttpUtils.parseResponse[WorkflowResponse](response, "WE")
      val res = Await.result(f, 1.second)
      res.statusUpdateDate should be (LocalDateTime.of(2021, 3, 14, 17, 2, 58, 426000000))
      res.statusUpdateDate.toInstant(ZoneOffset.UTC)
    }
  }

}
