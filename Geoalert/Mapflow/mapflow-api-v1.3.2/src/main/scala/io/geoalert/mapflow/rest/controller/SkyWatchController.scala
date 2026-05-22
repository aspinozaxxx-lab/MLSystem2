package io.geoalert.mapflow.rest.controller

import scala.concurrent.ExecutionContext
import scala.concurrent.Future

import akka.http.scaladsl.model.HttpResponse
import com.typesafe.scalalogging.LazyLogging
import io.circe._
import io.circe.parser._

import io.geoalert.mapflow.exception.ParsingError
import io.geoalert.mapflow.model.GetSkyWatchMetaInput
import io.geoalert.mapflow.model.SkyWatchAnswer
import io.geoalert.mapflow.model.SkyWatchAnswerData
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.providers.skywatch.SkyWatchCatalogClient
import io.geoalert.mapflow.rest.utils.ControllerConstants.SkyWatchControllerConstants._

//TODO: Move logic to SkyWatchCatalogService
class SkyWatchController(client: SkyWatchCatalogClient) extends LazyLogging {
  implicit val ec: ExecutionContext = ExecutionContext.global

  def getSkyWatchAnswerId(input: GetSkyWatchMetaInput)(user: User): Future[SkyWatchAnswer] = {
    logger.info(s"Downloading answer id from skywatch for user ${user.email}")

    for {
      answerWithId <- client.getSkyWatchMetaAnswerId(input)
      id = extractSkyWatchResponseId(input, answerWithId)
    } yield SkyWatchAnswer(SkyWatchAnswerData(id))
  }

  def getSkyWatchMetaPage(
      answerId: String,
      cursor: Option[String],
    )(
      user: User
    ): Future[HttpResponse] = {
    logger.info(s"Downloading meta page from skywatch for user ${user.email}")

    client.getSkyWatchMetaPage(answerId, cursor)
  }

  private def extractSkyWatchResponseId(input: GetSkyWatchMetaInput, answer: String): String = {
    val doc = parse(answer)
      .getOrElse(throw ParsingError(s"failed to parse answer from SkyWatch for input $input"))
    val cursor: HCursor = doc.hcursor

    cursor
      .downField(dataField)
      .downField(idField)
      .as[String]
      .toTry
      .get
  }
}

object SkyWatchController {
  def apply(): SkyWatchController = new SkyWatchController(new SkyWatchCatalogClient())
}
