package io.geoalert.rastertileserver.route

import akka.http.scaladsl.testkit.ScalatestRouteTest


import akka.http.scaladsl.model._
import akka.http.scaladsl.server._
import org.scalatest.Matchers
import org.scalatest.WordSpec
import org.scalatest.mock.MockitoSugar
import org.mockito.Mockito._

import io.geoalert.rastertileserver.cog.CogService

class RestSpec extends WordSpec with Matchers with ScalatestRouteTest with MockitoSugar {

  val mockCogService = mock[CogService]

  object TestObject extends CogRouteAbstract {
    val cogService = mockCogService
  }

  "The CogRoute" should {

    "Return 200:OK and image data if the image tile was retreived" in {
      when(mockCogService.getTile(1,1,1, "uri")).thenReturn(Array.fill[Byte](1024)(0))
        
      val getRequest = HttpRequest(HttpMethods.GET,
                                   uri = "api/v0/cogs/tiles/1/1/1.png")

      getRequest ~>  Route.seal(TestObject.apply()) ~> check {
        status.isSuccess() shouldEqual true
        responseAs[Array[Byte]] shouldEqual Array.fill[Byte](1024)(0)
      }
    }
  }
}


//    "Return 404: not found if provided s3 link does is invalid (no file or folder there)" in {
 //     when(cogImpl.getTile(2, "test")).thenReturn()
//
//    }

//    "Return 204: NO_CONTENT if the link is OK but the tile was not returned" in {      
//      when(cogImpl.getTile(2, "test")).thenReturn()

//    }
//  }
//class ExampleSpec extends AnyFlatSpec with should.Matchers {

  //"A List" should "do somthing" in {
 //   val list = List[Int](1, 2, 3)
  //  list.head should be (1)
 // }

 // it should "throw NoSuchElementException if an empty list is requested" in {
 //   val emptyList = List[Int]()
//    a [NoSuchElementException] should be thrownBy {
//      emptyList.head
//    } 
//  }
//}