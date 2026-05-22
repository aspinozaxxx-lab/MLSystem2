package io.geoalert.mapflow.rest

import akka.NotUsed
import akka.http.scaladsl.common.EntityStreamingSupport
import akka.http.scaladsl.marshalling.Marshaller
import akka.http.scaladsl.marshalling.Marshalling
import akka.http.scaladsl.model.ContentTypes
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.server.Route
import akka.http.scaladsl.server.directives.RouteDirectives
import akka.stream.scaladsl.Source
import akka.util.ByteString
import cats.data.EitherT
import cats.effect.IO
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import doobie.ConnectionIO
import doobie.syntax.connectionio._
import io.circe.Encoder

import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.Encoders

trait RestImplicits extends Encoders with RouteDirectives {
  implicit val responseAsCsv: Marshaller[List[String], ByteString] =
    Marshaller.strict[List[String], ByteString] { row =>
      Marshalling.WithFixedContentType(
        ContentTypes.`text/csv(UTF-8)`,
        () => ByteString(row.mkString(",")),
      )
    }

  implicit val csvStreaming = EntityStreamingSupport.csv()
  def toComplete[T](arg: ConnectionIO[T])(implicit encoder: Encoder[T]): Route = complete(
    arg.transact(xa).unsafeToFuture()
  )
  def completeAsCsv(arg: ConnectionIO[Source[List[String], NotUsed]]): Route = complete(
    arg.transact(xa).unsafeToFuture()
  )

  def toComplete[T](
      arg: EitherT[ConnectionIO, ApplicationError, T]
    )(implicit
      encoder: Encoder[T]
    ): Route = complete(
    arg
      .rethrowT
      .transact(xa)
      .unsafeToFuture()
  )

  def toComplete[T](arg: EitherT[ConnectionIO, ApplicationError, HttpResponse]): Route = complete(
    arg
      .rethrowT
      .transact(xa)
      .unsafeToFuture()
  )

  def toComplete[T](arg: IO[T])(implicit encoder: Encoder[T]): Route = complete(
    arg.unsafeToFuture()
  )

  def toComplete(arg: IO[HttpResponse]): Route = complete(arg.unsafeToFuture())

  def toComplete(arg: ConnectionIO[HttpResponse]): Route = complete(
    arg.transact(xa).unsafeToFuture()
  )
}
