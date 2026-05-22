package io.geoalert.mapflow.graphql.schema

import java.util.UUID

import io.circe.generic.auto._
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.EnumType
import sangria.schema.EnumValue
import sangria.schema.InputObjectType
import sangria.schema.ListInputType
import sangria.schema.ObjectType
import sangria.schema.OptionInputType
import sangria.schema.StringType

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.CreateUserInput
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.UpdateUserInput
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.UserBrief

trait UserSchema extends CommonSchema with DataProviderSchema {
  implicit val CreateUserInputType: InputObjectType[CreateUserInput] =
    deriveInputObjectType[CreateUserInput]()
  implicit val UpdateUserInputType: InputObjectType[UpdateUserInput] =
    deriveInputObjectType[UpdateUserInput]()

  implicit val RoleEnum: EnumType[Role] = EnumType(
    "Role",
    None,
    List(
      EnumValue(Role.Admin.repr, value = Role.Admin),
      EnumValue(Role.User.repr, value = Role.User),
    ),
  )

  implicit val BillingTypeEnum: EnumType[BillingType] = EnumType(
    "BillingType",
    None,
    List(
      EnumValue(BillingType.Area.repr, value = BillingType.Area),
      EnumValue(BillingType.Credits.repr, value = BillingType.Credits),
      EnumValue(BillingType.None.repr, value = BillingType.None),
    ),
  )

  implicit val UserType: ObjectType[GraphQLContext, User] = deriveObjectType[GraphQLContext, User]()
  implicit val UserBriefType: ObjectType[GraphQLContext, UserBrief] =
    deriveObjectType[GraphQLContext, UserBrief]()

  val CreateUserArg: Argument[CreateUserInput] = Argument("data", CreateUserInputType)
  val UpdateUserArg: Argument[UpdateUserInput] = Argument("data", UpdateUserInputType)

  val UserIdArg: Argument[UUID] = Argument("userId", UuidIdType)
  val UserIdsArg: Argument[Option[Seq[UUID]]] =
    Argument("userIds", OptionInputType(ListInputType(UuidIdType)))

  val UserEmailsArg: Argument[Option[Seq[String]]] =
    Argument("emails", OptionInputType(ListInputType(StringType)))
  val UserRolesArg: Argument[Option[Seq[Role]]] =
    Argument("roles", OptionInputType(ListInputType(RoleEnum)))
}
