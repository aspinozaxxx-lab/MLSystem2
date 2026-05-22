package io.geoalert.mapflow.exception

import scala.reflect.ClassTag

import akka.http.scaladsl.model.StatusCode
import akka.http.scaladsl.model.StatusCodes
import cats.data.NonEmptyChain
import io.geoalert.mapflow.implicits.CommonOps._

trait ApplicationError extends Exception {
  override def getMessage: String
  def code: String
  def statusCode: StatusCode
}

trait InternalError extends ApplicationError

trait UserError extends ApplicationError

case class ExternalSystemError(message: String) extends InternalError {
  override def getMessage: String = message
  override val code: String = "EXTERNAL_SYSTEM_ERROR"
  override val statusCode: StatusCode = StatusCodes.InternalServerError
}

case object ExternalSystemError {
  def apply(systemName: String, cause: Throwable): ExternalSystemError =
    ExternalSystemError(s"Error during $systemName call: $cause")

  def apply(
      systemName: String,
      statusCode: Int,
      body: String,
    ): ExternalSystemError =
    ExternalSystemError(s"$systemName returned status $statusCode: $body")
}

class AuthenticationError() extends ApplicationError {
  override val code = "AUTHENTICATION_ERROR"
  override def getMessage: String = "Wrong login/password."
  override val statusCode: StatusCode = StatusCodes.Unauthorized
}

case class AccessDenied(msg: String) extends UserError {
  override val code = "ACCESS_DENIED"
  override def getMessage: String = msg
  override val statusCode: StatusCode = StatusCodes.Forbidden
}

case class NoToken() extends AuthenticationError {
  override val code = "NO_TOKEN"
  override val getMessage: String = s"JWT token is not provided"
  override val statusCode: StatusCode = StatusCodes.Unauthorized
}

case class BadToken(msg: String) extends AuthenticationError {
  override val code = "BAD_TOKEN"
  override def getMessage: String = s"Invalid token: 
  override val statusCode: StatusCode = StatusCodes.BadRequest
}

case class TokenExpired(msg: String) extends AuthenticationError {
  override val code = "TOKEN_EXPIRED"
  override def getMessage: String = msg
  override val statusCode: StatusCode = StatusCodes.Unauthorized
}

class BadRequest(message: String, val code: String = "BAD_REQUEST") extends UserError {
  override def getMessage: String = message
  override val statusCode: StatusCode = StatusCodes.BadRequest
}

class Conflict(message: String, val code: String = "CONFLICT") extends UserError {
  override def getMessage: String = message
  override val statusCode: StatusCode = StatusCodes.Conflict
}

case class UserAlreadyExist(message: String) extends Conflict(message, "USER_ALREADY_EXIST")

object BadRequest {
  def apply(message: String, code: String = "BAD_REQUEST") = new BadRequest(message, code)
}

case object EmptySelection
    extends BadRequest(s"The selected Aois contain no data.", "EMPTY_SELECTION")

case object WdInUse
    extends BadRequest(s"This WorkflowDef is in use and can't be deleted.", "WD_IN_USE")

case class LoginTaken(email: String)
    extends BadRequest(s"User $email already exists.", "LOGIN_TAKEN")

case class FilePartMissing(name: String)
    extends BadRequest(s"Multipart part '$name' is missing", "FILE_PART_MISSING")

case class TooLargeProcessing(area: Long, aoiAreaLimit: Long)
    extends BadRequest(
      s"Processings larger than $aoiAreaLimit sq.m. are prohibited. Area: $area sq.m.",
      "TOO_LARGE_PROCESSING",
    )

case class AreaLimitExceeded(processed: Long, limit: Long)
    extends BadRequest(
      s"An accumulative area limit of $limit sq.m. is set for the user. " +
        s"Current operation would increase the total accumulative processed area to $processed sq.m.",
      "AREA_LIMIT_EXCEEDED",
    )

case class ParsingError(cause: Throwable, message: String)
    extends BadRequest(message, "WORKFLOW_DEF_PARSING_ERROR") {
  override def getCause = cause
}
case class MapfileParseError(cause: Option[Throwable], message: String)
    extends BadRequest(message, "MAPFILE_PARSE_ERROR") {
  override def getCause: Throwable = cause.getOrElse(super.getCause)
}
case class MapfileFetchError(cause: Throwable, message: String)
    extends BadRequest(message, "MAPFILE_FETCH_ERROR") {
  override def getCause: Throwable = cause
}

object ParsingError {
  def apply(message: String): ParsingError = ParsingError(null, message)
  def apply(cause: Throwable): ParsingError = ParsingError(cause, cause.getMessage)
}

case class ValidationError(errors: NonEmptyChain[Throwable])
    extends BadRequest(
      "The following problems were found:" + errors
        .toChain
        .toList
        .distinct
        .map("\n" + _.getMessage)
        .shortString(10)
    ) {}

object ValidationError {
  def apply(error: Throwable): ValidationError = ValidationError(NonEmptyChain(error))
}

case class NotFound(message: String) extends BadRequest(message, "NOT_FOUND") {
  override val statusCode: StatusCode = StatusCodes.NotFound
}

object NotFound {
  def apply[A](id: Any)(implicit A: ClassTag[A]): NotFound =
    NotFound(s"Entity ${A.runtimeClass.getSimpleName} with id=$id not found.")

  def apply[A](ids: List[Any])(implicit A: ClassTag[A]): NotFound =
    NotFound(
      s"Entities ${A.runtimeClass.getSimpleName} with the following ids were not found: ${ids.mkString(", ")}"
    )
}

case class Forbidden(msg: String) extends UserError {
  override def getMessage: String = msg
  override val statusCode: StatusCode = StatusCodes.Forbidden
  override val code: String = "FORBIDDEN"
}

case class InternalServerError(msg: String) extends InternalError {
  override def getMessage: String = msg
  override val code = "INTERNAL_ERROR"
  override val statusCode: StatusCode = StatusCodes.InternalServerError
}

object InternalServerError {
  def apply(message: String) = new InternalServerError(message)
  def apply(cause: Throwable) = new InternalServerError(cause.getMessage)
}
