package io.geoalert.mapflow.implicits

import java.time.Instant
import java.util.Date
import cats.syntax.either._
import doobie.postgres.implicits._
import doobie.util.meta.Meta
import io.circe.Json
import io.circe.parser.parse
import io.geoalert.mapflow.model.{BillingType, RequiredAction, ReviewStatus, Status, TeamMemberRole}
import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole
import io.geoalert.mapflow.model.enums.MemberRole
import io.getquill.MappedEncoding
import org.postgresql.util.PGobject

object Postgres {
  implicit val jsonMeta: Meta[Json] =
    Meta
      .Advanced
      .other[PGobject]("json")
      .timap[Json](a => parse(a.getValue).leftMap[Json](e => throw e).merge) { a =>
        val o = new PGobject
        o.setType("json")
        o.setValue(a.noSpaces)
        o
      }

  implicit val instantDecoder: MappedEncoding[Date, Instant] =
    MappedEncoding(_.toInstant)

  implicit val instantEncoder: MappedEncoding[Instant, Date] =
    MappedEncoding(Date.from)

  implicit val jsonDecoder: MappedEncoding[String, Json] =
    MappedEncoding[String, Json](s => io.circe.jawn.parse(s).fold(e => throw e, j => j))

  implicit val jsonEncoder: MappedEncoding[Json, String] =
    MappedEncoding[Json, String](_.noSpaces)

  implicit val RequiredActionMeta: Meta[RequiredAction] =
    pgJavaEnum[RequiredAction]("workflow_required_action")

  implicit val teamMemberRoleMeta: Meta[TeamMemberRole] = Meta[String]
    .timap(str =>
      TeamMemberRole
        .fromString(str)
        .getOrElse(throw new IllegalStateException(s"Unexpected team member role $str"))
    )(_.toString)

  implicit val status: Meta[Status] = Meta[String].timap(str => Status.fromString(str))(_.repr)

  implicit val MemberRoleMeta: Meta[MemberRole] =
    pgEnumString[MemberRole](
      "member_role",
      str =>
        MemberRole
          .withNameOption(str)
          .getOrElse(throw new IllegalStateException(s"Unexpected member role $str")),
      _.entryName,
    )
  implicit val billingTypeMeta: Meta[BillingType] = Meta[String]
    .timap(str => BillingType.fromString(str))(_.repr)

  implicit val reviewStatusMeta: Meta[ReviewStatus] = Meta[String]
    .timap(str => ReviewStatus.fromString(str))(_.repr)
}
