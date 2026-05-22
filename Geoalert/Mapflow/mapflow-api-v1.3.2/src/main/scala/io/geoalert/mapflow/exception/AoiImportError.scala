package io.geoalert.mapflow.exception

import akka.http.scaladsl.model.StatusCode
import akka.http.scaladsl.model.StatusCodes
import cats.data.NonEmptyChain

sealed trait AoiImportError extends ApplicationError {
  override val code = "AOI_IMPORT_ERROR"
}

sealed trait Critical

sealed trait NonCritical

case class AccumulativeAoiImportError(msg: String)
    extends InternalError
       with AoiImportError
       with Critical {
  override def getMessage: String = msg
  override val statusCode: StatusCode = StatusCodes.InternalServerError
}

object AccumulativeAoiImportError {
  def apply(errors: NonEmptyChain[Throwable]): AccumulativeAoiImportError = {
    val msg = errors
      .toChain
      .toList
      .distinct
      .map(_.getMessage)
      .foldLeft("The following problems were found:")(_ + "\n" + _)
    AccumulativeAoiImportError(msg)
  }
}

case class GeometryError(msg: String) extends InternalError with AoiImportError with Critical {
  override def getMessage: String = msg
  override val statusCode: StatusCode = StatusCodes.InternalServerError
}

object GeometryError {
  def apply(cause: Throwable): GeometryError = GeometryError(s"Error importing Aoi: $cause")
}

case object TooSmallGeometry
    extends BadRequest("Too small geometry.")
       with AoiImportError
       with NonCritical
