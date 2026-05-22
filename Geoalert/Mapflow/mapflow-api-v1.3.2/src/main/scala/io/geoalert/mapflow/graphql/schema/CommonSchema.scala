package io.geoalert.mapflow.graphql.schema

import java.time.Instant
import java.time.format.DateTimeFormatter
import java.util.UUID

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import cats.syntax.either._
import io.circe.Decoder
import io.circe.Encoder
import io.circe.Json
import io.circe.parser.parse
import io.geoalert.mapflow.model.DataSource
import io.geoalert.mapflow.model.SourceType
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.enums.MemberRole
import sangria.ast
import sangria.macros.derive.TransformValueNames
import sangria.macros.derive.deriveEnumType
import sangria.marshalling.circe._
import sangria.schema._
import sangria.util.StringUtil
import sangria.validation.ValueCoercionViolation

trait CommonSchema {
  case object DateCoercionViolation extends ValueCoercionViolation("Date value expected")

  private def parseInstant(s: String) = Try(Instant.parse(s)) match {
    case Success(date) => Right(date)
    case Failure(_) => Left(DateCoercionViolation)
  }

  private val dateTimeFormatter = DateTimeFormatter.ISO_INSTANT

  implicit val StatusEnum: EnumType[Status] =
    deriveEnumType[Status](TransformValueNames(StringUtil.camelCaseToUnderscore(_).toUpperCase))

  implicit val MemberRoleEnum: EnumType[MemberRole] =
    deriveEnumType[MemberRole](
      TransformValueNames(StringUtil.camelCaseToUnderscore)
    )

  implicit val InstantType: ScalarType[Instant] = ScalarType[Instant](
    "Date",
    coerceOutput = (i, _) => dateTimeFormatter.format(i),
    coerceUserInput = {
      case s: String => parseInstant(s)
      case _ => Left(DateCoercionViolation)
    },
    coerceInput = {
      case sangria.ast.StringValue(s, _, _, _, _) => parseInstant(s)
      case _ => Left(DateCoercionViolation)
    },
  )
  private case object JsonCoercionViolation extends ValueCoercionViolation("Invalid JSON value.")

  implicit val JsonType: ScalarType[Json] =
    ScalarType[Json](
      "Json",
      description = Some("String encoded JSON value"),
      coerceOutput = (value, _) => {
        val result =
          if (value.asObject.exists(_.isEmpty)) Json.Null
          else value

        result.spaces2
      },
      coerceUserInput = {
        case v: String => parse(v).leftMap(_ => JsonCoercionViolation)
        case _ => Left(JsonCoercionViolation)
      },
      coerceInput = {
        case ast.StringValue(jsonStr, _, _, _, _) =>
          parse(jsonStr).leftMap(_ => JsonCoercionViolation)
        case _ =>
          Left(JsonCoercionViolation)
      },
    )

  case object UuidCoercionViolation extends ValueCoercionViolation("UUID value expected")

  implicit val UuidIdType: ScalarAlias[UUID, String] = ScalarAlias[UUID, String](
    IDType,
    _.toString,
    s => Try(UUID.fromString(s)).toEither.leftMap(_ => UuidCoercionViolation),
  )

  val IdArg: Argument[UUID] = Argument("id", UuidIdType)
  val IdsArg: Argument[Option[Seq[UUID]]] =
    Argument("ids", OptionInputType(ListInputType(UuidIdType)))

  val OffsetArg: Argument[Option[Int]] = Argument("offset", OptionInputType(IntType))
  val LimitArg: Argument[Option[Int]] = Argument("limit", OptionInputType(IntType))

  implicit val SourceTypeType: EnumType[SourceType.Value] = deriveEnumType[SourceType.Value]()
  implicit val DataSourceType: EnumType[DataSource.Value] = deriveEnumType[DataSource.Value]()

  implicit val sourceTypeDecoder: Decoder[SourceType.Value] = Decoder.decodeEnumeration(SourceType)
  implicit val sourceTypeEncoder: Encoder[SourceType.Value] = Encoder.encodeEnumeration(SourceType)

  implicit val dataSourceDecoder: Decoder[DataSource.Value] = Decoder.decodeEnumeration(DataSource)
  implicit val dataSourceEncoder: Encoder[DataSource.Value] = Encoder.encodeEnumeration(DataSource)
}
