package io.geoalert.mapflow.util

import java.io.InputStream
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.concurrent.duration._

import akka.stream.scaladsl.Source
import akka.stream.scaladsl.StreamConverters
import akka.util.ByteString
import cats.data.EitherT
import cats.data.OptionT
import cats.effect.ContextShift
import cats.effect.IO
import cats.instances.future._
import cats.instances.option._
import cats.syntax.either._
import cats.syntax.traverse._
import com.google.common.util.concurrent.ThreadFactoryBuilder
import doobie._
import doobie.implicits._

import io.geoalert.mapflow.AkkaSystem._
import io.geoalert.mapflow.exception.FilePartMissing
import io.geoalert.mapflow.graphql.GraphQLContext

object UploadUtils {
  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        8,
        new ThreadFactoryBuilder().setNameFormat("uploads-%d").build(),
      )
    )
  )

  private def sourceOption(ctx: GraphQLContext, file: String): Option[Source[ByteString, Any]] =
    ctx.files.get(file)

  // TODO lazy streaming
  def streamOption(ctx: GraphQLContext, file: String): Option[InputStream] =
    // This is still not true streaming because HttpServer has '.toStrict'.
    // This is better though then doing a fold to an Array[Byte] - less memory footprint.
    sourceOption(ctx, file).map(_.runWith(StreamConverters.asInputStream(15.minutes)))

  def streamEither(ctx: GraphQLContext, file: String): Either[FilePartMissing, InputStream] =
    Either.fromOption(streamOption(ctx, file), FilePartMissing(file))

  def strictOption(ctx: GraphQLContext, file: String): OptionT[Future, Array[Byte]] = {
    def fileContents(s: Source[ByteString, Any]) =
      s.runFold(ByteString.empty)(_ ++ _).map(_.toArray)
    OptionT(sourceOption(ctx, file).map(fileContents).sequence)
  }

  def strictEither(
      ctx: GraphQLContext,
      file: String,
    ): EitherT[Future, FilePartMissing, Array[Byte]] =
    EitherT.fromOptionF(strictOption(ctx, file).value, FilePartMissing(file))

  def strictEitherIO(
      ctx: GraphQLContext,
      file: String,
    ): EitherT[ConnectionIO, FilePartMissing, Array[Byte]] = {
    val io = IO.fromFuture(IO(strictEither(ctx, file).value))
    EitherT(io.to[ConnectionIO])
  }
}
