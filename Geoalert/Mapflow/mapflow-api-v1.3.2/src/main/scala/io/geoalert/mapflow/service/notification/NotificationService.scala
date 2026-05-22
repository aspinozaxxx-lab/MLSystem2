package io.geoalert.mapflow.service.notification

import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.UUID
import java.util.concurrent.Executors

import scala.collection.mutable
import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor

import cats.effect.ContextShift
import cats.effect.IO
import cats.implicits.catsSyntaxApplicativeId
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO

import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.SourceType
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.repo.ProcessingDto
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.UserDto
import io.geoalert.mapflow.repo.UserRepo
import io.geoalert.mapflow.service.RasterService

class NotificationService(rasterService: RasterService) extends LazyLogging {
  /** Interact with Telegram API in separate thread
    */
  implicit private val ec: ExecutionContextExecutor =
    ExecutionContext.fromExecutor(Executors.newFixedThreadPool(1))
  implicit private val cs: ContextShift[IO] = IO.contextShift(ec)

  // TODO: This set is not thread safe
  val notifiedProcessingsSet: mutable.Set[UUID] = mutable.Set[UUID]()

  def sendFailedProcessingNotification(
      processing: ProcessingDto,
      wfId: UUID,
      wfExternalId: Option[String],
      oldStatus: Status,
      newStatus: Status,
      isUpdate: Boolean,
      messages: List[Message],
    ): ConnectionIO[Unit] = {
    logger.info(
      s"Sending notification on failed processing: processing ${processing.id} ${if (processing.archived) "(archived)"
        else ""} changed from $oldStatus to $newStatus"
    )
    if (
        enableTelegramNotificationAboutFailedProcessings &
          (oldStatus.intVal != Status.Failed.intVal) & (newStatus.intVal == Status.Failed.intVal) &
          (!notifiedProcessingsSet.contains(processing.id)) & isUpdate & !processing.archived
    )
      for {
        project <- ProjectRepo.getProject(processing.projectId).rethrowT
        user <- UserRepo.getUser(project.userId).rethrowT
        _ <- formMessageAndSend(wfId, wfExternalId, processing, user, messages)
      } yield {}
    else
      {}.pure[ConnectionIO]
  }

  private def formMessageAndSend(
      wfId: UUID,
      wfExternalId: Option[String],
      processing: ProcessingDto,
      user: UserDto,
      messages: List[Message],
    ): ConnectionIO[Unit] = {
    val sb = new StringBuilder
    val formatter: DateTimeFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME

    notifiedProcessingsSet.add(processing.id)

    val params = ProcessingParams(processing.params)
    val meta = ProcessingMeta(processing.meta)

    val link = s"$externalUrl/projects/${processing.projectId}/processings/${processing.id}"
    sb.append(s"Processing `${processing.id}` failed.\n")
    sb.append(s"Workflow: `$wfId`")
    wfExternalId.foreach(id => sb.append(s" ($id)"))
    sb.append("\n")
    sb.append(s"Application: `${meta.sourceApp.getOrElse("API")}`\n")
    if (processing.sourceType.contains(SourceType.local.toString) && params.url.isDefined) {
      val downloadUrl = params.url.map(rasterService.generatePresignedUrl)
      sb.append(s"Minio File: ${downloadUrl.getOrElse("")}\n")
    }
    sb.append(s"Link: $link\n")
    sb.append(s"Environment: `$environment`\n")
    sb.append(s"User email: `${user.email}`\n")

    sb.append(
      s"Created: `${formatter.format(ZonedDateTime.ofInstant(processing.created, ZoneId.of("UTC+3")))}`\n"
    )
    sb.append(s"Updated: `${formatter.format(ZonedDateTime.now(ZoneId.of("UTC+3")))}`\n")

    val messagesString: String = messages.map(m => escapeTgMessage(m.message)).mkString("\n")

    if (messagesString.nonEmpty)
      sb.append(s"Messages: \n$messagesString\n")

    val codes = messages
      .map(m => escapeTgMessage(m.code))
      .distinct
      .mkString("\n")

    if (codes.nonEmpty)
      sb.append(s"Error codes:\n$codes\n")

    val future = Telegram
      .sendMessage(sb.mkString)
      .unsafeToFuture()
      .recover {
        case e: Throwable =>
          logger.error(s"Telegram notification about failed processing failed. Workflow: $wfId", e)
      }

    IO.fromFuture(IO(future)).to[ConnectionIO]
  }

  private def escapeTgMessage(msg: String): String = {
    val escaped = msg.replaceAll("`", "\\`")
    s"`$escaped`"
  }

  def removeRestartingProcessing(processingId: UUID): Unit =
    notifiedProcessingsSet.remove(processingId)
}

object NotificationService {
  def apply(rasterService: RasterService): NotificationService = new NotificationService(
    rasterService
  )
}
