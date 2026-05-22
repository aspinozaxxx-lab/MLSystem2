package io.geoalert.mapflow.graphql

import akka.http.javadsl.model.headers.SameSite
import akka.http.scaladsl.model.headers.HttpCookie
import akka.http.scaladsl.server.Directive0
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.directives.BasicDirectives
import akka.stream.scaladsl.Source
import akka.util.ByteString
import com.typesafe.scalalogging.LazyLogging

import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User

case class GraphQLContext(files: Map[String, Source[ByteString, Any]], user: User)
    extends LazyLogging {

  // This is a bit hacky.
  // Mutable object that stores whether a cookie needs to be set or deleted in a http response.
  // May be mutated during graphql query execution.
  val authCookieDirective = new AuthCookieDirective

  def checkUser(): Unit =
    if (user.role != Role.Admin) {
      logger.error(
        s"Access denied for ${user.email} to GraphQL. Only administrators allowed to use GraphQL"
      )
      throw AccessDenied(s"Only administrators allowed to use GraphQL")
    }
}

class AuthCookieDirective {
  private var directive: Directive0 = BasicDirectives.mapInnerRoute(r => r)

  def set(token:  Unit =
    directive = setCookie(
      HttpCookie("token", token).withSecure(true).withHttpOnly(true).withSameSite(SameSite.None)
    )

  def delete(): Unit =
    directive = deleteCookie("token")

  def getDirective: Directive0 = directive
}
