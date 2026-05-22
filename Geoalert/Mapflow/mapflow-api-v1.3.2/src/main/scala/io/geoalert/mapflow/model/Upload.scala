package io.geoalert.mapflow.model

import java.io.InputStream

import scala.concurrent.Future

import cats.data.EitherT
import cats.data.OptionT
import doobie.ConnectionIO

import io.geoalert.mapflow.exception.FilePartMissing
import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.util.UploadUtils

case class Upload(part: String) {
  def streamEither(ctx: GraphQLContext): Either[FilePartMissing, InputStream] =
    UploadUtils.streamEither(ctx, part)

  def streamOption(ctx: GraphQLContext): Option[InputStream] =
    UploadUtils.streamOption(ctx, part)

  def strictEither(ctx: GraphQLContext): EitherT[Future, FilePartMissing, Array[Byte]] =
    UploadUtils.strictEither(ctx, part)

  def strictOption(ctx: GraphQLContext): OptionT[Future, Array[Byte]] =
    UploadUtils.strictOption(ctx, part)

  def strictEitherIO(ctx: GraphQLContext): EitherT[ConnectionIO, FilePartMissing, Array[Byte]] =
    UploadUtils.strictEitherIO(ctx, part)

  override def toString: String = "[file]"
}
