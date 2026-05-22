package io.geoalert.mapflow.graphql

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.util.Failure
import scala.util.Success

import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.Json
import io.circe.generic.auto._
import io.circe.parser

import io.geoalert.mapflow.AkkaSystem._
import io.geoalert.mapflow.rest.Authorization

object GraphQLRoute extends Directives with Authorization {
  def executeGraphQlWithJson: Route = (path("graphql") & post & authorized & entity(as[Json])) {
    (user, js) =>
      GraphQLServer.endpoint(GraphQLRequestData(js, user))
  }

  def executeGraphQlWithForm: Route =
    (path("graphql") & post & authorized & entity(as[Multipart.FormData])) { (user, formData) =>
      val requestData = formData.parts.runFoldAsync(GraphQLRequestData(Json.obj(), user)) {
        case (d, p) if p.name == "operations" =>
          p.toStrict(10.seconds)
            .map(s => parser.parse(s.entity.data.utf8String))
            .map(js => d.copy(request = js.toTry.get))
        case (d, p) if p.name == "map" =>
          p.toStrict(10.seconds)
            .map(s => parser.decode[Map[String, List[String]]](s.entity.data.utf8String))
            .map(js => d.copy(map = Some(js.toTry.get)))
        case (d, p) =>
          p.toStrict(5.minutes)
            .map(s => d.copy(files = d.files + (p.name -> s.entity.dataBytes)))
      }
      onComplete(requestData) {
        case Success(d) => GraphQLServer.endpoint(d)
        case Failure(e) => failWith(e)
      }
    }

  def getIndexHtml: Route = (path("graphql") & get & authorized) { user =>
      getFromResource("graphiql.html")
  }

  val routes: Route = concat(
    executeGraphQlWithJson,
    executeGraphQlWithForm,
    getIndexHtml,
  )
}
