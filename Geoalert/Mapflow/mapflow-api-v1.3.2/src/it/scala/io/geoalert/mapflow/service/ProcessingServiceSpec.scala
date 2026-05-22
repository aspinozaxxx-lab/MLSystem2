package io.geoalert.mapflow.service

import scala.concurrent.Await
import scala.concurrent.ExecutionContext
import scala.concurrent.duration._
import scala.util.Failure
import scala.util.Success
import scala.util.Try

import cats.syntax.option._
import doobie.implicits._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.Forbidden
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.ProjectUtil
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class ProcessingServiceSpec extends DbIntegrationTest with Services {
  implicit val ec: ExecutionContext = ExecutionContext.global

  describe("ProcessingService") {
    it(s"shouldn't show other users processings") {
      val user1 = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val user2 = UserUtil.createUser("user2@example.com", None, "pwd123456".some)
      val prc = ProcessingUtil.createProcessing(user1)

      processingService
        .getProcessing(prc.id)(user2)
        .value
        .transact(xa)
        .unsafeRunSync() should matchPattern {
        case Left(NotFound(_)) =>
      }

      processingService
        .getProcessing(prc.id)(user1)
        .value
        .transact(xa)
        .unsafeRunSync() should matchPattern {
        case Right(_) =>
      }

      val updInput = UpdateProcessingInput(prc.id, name = "x".some)
      val updTry = Try(
        Await.result(
          processingService.updateProcessing(updInput)(user2).transact(xa).unsafeToFuture(),
          2.seconds,
        )
      )
      updTry should matchPattern { case Failure(Forbidden(_)) => }
    }

    it(s"should show all processings of admin") {
      val user1 = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val prc = ProcessingUtil.createProcessing(user1)

      processingService
        .getProcessing(prc.id)(UserUtil.admin)
        .value
        .transact(xa)
        .unsafeRunSync() should matchPattern {
        case Right(_) =>
      }

      val updInput = UpdateProcessingInput(prc.id, name = "x".some)
      val updTry = Try(
        Await.result(
          processingService
            .updateProcessing(updInput)(user1)
            .transact(xa)
            .unsafeToFuture(),
          2.seconds,
        )
      )
      updTry should matchPattern { case Success(_) => }
    }
  }

  describe("CRUD methods") {
    it("should create processing") {
      val regularUser = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val project = ProjectUtil.createProject(regularUser)

      val wdId = WorkflowDefUtil.createWd()
      val future = processingService
        .createProcessing(
          CreateProcessingInput(
            project.id,
            None,
            None,
            wdId,
            None,
            "test processing".some,
            None,
            42,
            None,
            None,
            None,
          )
        )(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeToFuture()

      val processing = Await.result(future, 2.seconds)
      processing.name should be("test processing")
    }

    it("should retrieve processing") {
      val regularUser = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val processing = ProcessingUtil.createProcessing(regularUser)
      val future = processingService
        .getProcessing(processing.id)(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeToFuture()
      val prc = Await.result(future, 2.seconds)
      prc.id should be(processing.id)
    }

    it("should update processing") {
      val regularUser = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val processing = ProcessingUtil.createProcessing(regularUser)
      val future = processingService
        .updateProcessing(
          UpdateProcessingInput(
            processing.id,
            name = "new name".some,
            description = "new description".some,
            cost = 40L.some,
          )
        )(regularUser)
        .transact(xa)
        .unsafeToFuture()
      val prc = Await.result(future, 2.seconds)
      prc.name should be("new name")
      prc.cost should be(40.0.some)
    }

    it("should delete processing") {
      val regularUser = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val processing = ProcessingUtil.createProcessing(regularUser)
      val future = processingService
        .archiveProcessing(processing.id)(regularUser)
        .transact(xa)
        .unsafeToFuture()
      val result = Await.result(future, 2.seconds)
      result should be("OK")

      val future2 = processingService
        .getProcessing(processing.id)(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeToFuture()
      Try(Await.result(future2, 2.seconds)) should matchPattern {
        case Failure(_: NotFound) =>
      }

      val future4 = processingService
        .getProcessingsWithArchived(List(processing.id))(UserUtil.admin)
        .transact(xa)
        .unsafeToFuture()
      val processings = Await.result(future4, 2.seconds)
      val ids = processings.map(_.id)
      ids should contain(processing.id)
    }
  }
}
