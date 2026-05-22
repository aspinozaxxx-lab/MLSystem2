package io.geoalert.mapflow.syntax

import java.net.URI
import java.net.URL

import cats.MonadThrow
import cats.data.EitherT
import io.circe._
import io.circe.parser.decode
trait CirceSyntax {
  implicit def circeSyntaxDecoderOps(s: String): DecoderOps = new DecoderOps(s)
  implicit def circeSyntaxJsonDecoderOps(json: Json): JsonDecoderOps = new JsonDecoderOps(json)
  implicit val urlEncoder: Encoder[URL] =
    Encoder.encodeString.contramap(_.toString)
  implicit val urlDecoder: Decoder[URL] =
    Decoder.decodeString.map(URI.create(_).toURL)
}

final class DecoderOps(private val s: String) {
  def decodeAs[A: Decoder]: A = decode[A](s).fold(throw _, json => json)
  def decodeAsF[F[_]: MonadThrow, A: Decoder]: F[A] =
    EitherT.fromEither[F](decode[A](s)).rethrowT
}
final class JsonDecoderOps(json: Json) {
  def decodeAs[A](implicit decoder: Decoder[A]): A =
    decoder.decodeJson(json).fold(throw _, json => json)
  def decodeAsF[F[_]: MonadThrow, A](implicit decoder: Decoder[A]): F[A] =
    EitherT.fromEither[F](decoder.decodeJson(json)).rethrowT
}
