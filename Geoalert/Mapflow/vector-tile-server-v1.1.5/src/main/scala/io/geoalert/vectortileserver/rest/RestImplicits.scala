package io.geoalert.vectortileserver.rest

import akka.http.scaladsl.server.directives.RouteDirectives
import akka.http.scaladsl.server.Route
import cats.effect.IO
import doobie.ConnectionIO
import doobie.syntax.connectionio._
import io.circe.Encoder
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._

import io.geoalert.vectortileserver.Db

trait RestImplicits extends RouteDirectives with Db {
  def toComplete[T](arg: ConnectionIO[T])(implicit encoder: Encoder[T]): Route = complete(arg.transact(xa).unsafeToFuture())

  def toComplete[T](arg: IO[T])(implicit encoder: Encoder[T]): Route = complete(arg.unsafeToFuture())
}