package io.geoalert.rastertileserver.exception

import akka.http.scaladsl.model.{StatusCode, StatusCodes}

sealed trait ApplicationError extends Exception with Serializable {
  def getMessage: String
  def code: String
  def statusCode: StatusCode
}

trait InternalError extends ApplicationError

trait UserError extends ApplicationError

case class InternalServerError(message: String) extends InternalError {
  override def getMessage: String = message
  override val code = "INTERNAL_ERROR"
  override val statusCode: StatusCode = StatusCodes.InternalServerError
}

case class NotFound(message: String) extends UserError() {
  override def getMessage: String = message
  override val code = "NOT_FOUND"
  override val statusCode: StatusCode = StatusCodes.NotFound
}

case class BadRequest(message: String) extends UserError() {
  override def getMessage: String = message
  override val code = "BAD_REQUEST"
  override val statusCode: StatusCode = StatusCodes.BadRequest
}
