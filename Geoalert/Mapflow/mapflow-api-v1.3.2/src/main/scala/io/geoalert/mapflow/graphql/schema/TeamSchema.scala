package io.geoalert.mapflow.graphql.schema

import java.time.Instant
import java.util.UUID

import io.circe.generic.auto._
import sangria.macros.derive.IncludeValues
import sangria.macros.derive.deriveEnumType
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.EnumType
import sangria.schema.InputObjectType
import sangria.schema.LongType
import sangria.schema.ObjectType
import sangria.schema.OptionInputType

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model.CreateTeamInput
import io.geoalert.mapflow.model.Team
import io.geoalert.mapflow.model.TeamMember
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole
import io.geoalert.mapflow.model.TeamWithMembers
import io.geoalert.mapflow.model.UpdateTeamInput

trait TeamSchema extends CommonSchema {
  implicit val TeamType: ObjectType[GraphQLContext, Team] = deriveObjectType[GraphQLContext, Team]()
  implicit val TeamMemberType: ObjectType[GraphQLContext, TeamMember] =
    deriveObjectType[GraphQLContext, TeamMember]()
  implicit val TeamWithMembersType: ObjectType[GraphQLContext, TeamWithMembers] =
    deriveObjectType[GraphQLContext, TeamWithMembers]()
  implicit val CreateTeamInputType: InputObjectType[CreateTeamInput] =
    deriveInputObjectType[CreateTeamInput]()
  implicit val UpdateTeamInputType: InputObjectType[UpdateTeamInput] =
    deriveInputObjectType[UpdateTeamInput]()

  implicit val TeamMemberRoleEnum: EnumType[TeamMemberRole] = deriveEnumType[TeamMemberRole](
    IncludeValues("MEMBER", "OWNER")
  )

  val CreateTeamArg: Argument[CreateTeamInput] = Argument("data", CreateTeamInputType)
  val UpdateTeamArg: Argument[UpdateTeamInput] = Argument("data", UpdateTeamInputType)

  val TeamIdArg: Argument[UUID] = Argument("teamId", UuidIdType)

  val TeamMemberRoleArg: Argument[TeamMemberRole] = Argument("role", TeamMemberRoleEnum)

  val ActiveUntilArg: Argument[Option[Instant]] =
    Argument("active_until", OptionInputType(InstantType))

  val AreaLimitArg: Argument[Option[Long]] = Argument("area_limit", OptionInputType(LongType))

  val CreditsLimitArg: Argument[Option[Long]] = Argument("credits_limit", OptionInputType(LongType))
}
