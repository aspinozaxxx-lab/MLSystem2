package io.geoalert.mapflow.service.notification

import java.net.URLEncoder
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor

import akka.http.scaladsl.client.RequestBuilding.Post
import akka.stream.scaladsl.Sink
import akka.stream.scaladsl.Source
import cats.effect.ContextShift
import cats.effect.IO
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging

import io.geoalert.mapflow.AkkaSystem._
import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.util.HttpUtils

object Telegram extends LazyLogging {
  private val reservedChars = List(
    '_', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!', '(', ')',
  )

  implicit val executionContext: ExecutionContextExecutor = ExecutionContext.global

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        8,
        new ThreadFactoryBuilder().setNameFormat("tg-client-%d").build(),
      )
    )
  )

  private val connectionFlow = HttpUtils.httpConnection(telegramApiUrl)

  def sendMessage(message: String): IO[Unit] = {
    logger.info(s"Sending message to telegram: $message")

    val escaped = reservedChars.foldLeft(message) { case (m, c) => m.replace(c.toString, "\\" + c) }
    val encoded = URLEncoder.encode(escaped, "UTF-8")
    val uri =
      s"/bot$telegramToken/sendMessage?chat_id=$telegramChatId&parse_mode=MarkdownV2&text=$encoded"

    def send() = Source
      .single(Post(uri))
      .via(connectionFlow)
      .runWith(Sink.head)

    val f = (for {
      r <- send()
      _ <- HttpUtils.parseResponseAsString(r, "Telegram")
    } yield ()).recover {
      case e: Throwable =>
        logger.error(s"Error sending telegram message\n$message", e)
        ()
    }

    IO.fromFuture(IO(f))
  }
}
