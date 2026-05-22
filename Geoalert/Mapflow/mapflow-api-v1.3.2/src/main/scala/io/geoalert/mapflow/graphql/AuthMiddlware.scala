package io.geoalert.mapflow.graphql

import scala.concurrent.Future

import cats.data.NonEmptyList
import cats.implicits.catsSyntaxOptionId
import cats.implicits.none
import sangria.execution.BeforeFieldResult
import sangria.execution.FieldTag
import sangria.execution.Middleware
import sangria.execution.MiddlewareBeforeField
import sangria.execution.MiddlewareQueryContext
import sangria.schema.Context
import sangria.schema.FutureValue

import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.model.Role

case object Authorized extends FieldTag
case class PrivilegeRequired(roles: NonEmptyList[Role]) extends FieldTag
object PrivilegeRequired {
  def apply(head: Role, tail: Role*): PrivilegeRequired =
    PrivilegeRequired(NonEmptyList(head, tail.toList))
}
object AuthMiddleware
    extends Middleware[GraphQLContext]
       with MiddlewareBeforeField[GraphQLContext] {
  override type QueryVal = Unit
  override type FieldVal = Unit

  override def beforeQuery(context: MiddlewareQueryContext[GraphQLContext, _, _]): QueryVal = ()

  override def afterQuery(
      queryVal: QueryVal,
      context: MiddlewareQueryContext[GraphQLContext, _, _],
    ): Unit = ()

  override def beforeField(
      queryVal: QueryVal,
      mctx: MiddlewareQueryContext[GraphQLContext, _, _],
      ctx: Context[GraphQLContext, _],
    ): BeforeFieldResult[GraphQLContext, FieldVal] =
    accessAllowed(ctx).fold(continue) { message =>
      overrideAction(FutureValue(Future.failed(AccessDenied(message))))
    }

  private def accessAllowed(ctx: Context[GraphQLContext, _]): Option[String] = {
    val privilegeRequired = ctx.field.tags.flatMap {
      case PrivilegeRequired(privileges) => privileges.toList
      case _ => Nil
    }
    if (privilegeRequired.nonEmpty && !privilegeRequired.contains(ctx.ctx.user.role))
      "User does not have the required privilege".some
    else none[String]
  }
}
