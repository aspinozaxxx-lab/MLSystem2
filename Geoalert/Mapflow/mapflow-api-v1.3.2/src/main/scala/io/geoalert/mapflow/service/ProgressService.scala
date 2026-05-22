package io.geoalert.mapflow.service

import java.util.UUID

import scala.concurrent.duration._

import cats.syntax.applicative._
import com.github.blemale.scaffeine.Cache
import com.github.blemale.scaffeine.Scaffeine
import doobie.ConnectionIO
import io.geoalert.mapflow.model.Progress
import io.geoalert.mapflow.model.Progress.lastUpdateDate
import io.geoalert.mapflow.model.ProjectProgress
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.repo.AoiDto
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.repo.ProcessingDto
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.repo.ProgressRepo
import io.geoalert.mapflow.repo.ProjectDto
import io.geoalert.mapflow.repo.ProjectRepo

/** Calculating AOI / Processing / Project progress
  */
class ProgressService {
  val cache: Cache[UUID, Progress] = Scaffeine()
    .expireAfterWrite(2.minutes)
    .maximumSize(1000)
    .build()

  // TODO: Consider managing Project <- Processing <- Aoi in this service
  def invalidateCache(
      projectIds: List[UUID],
      processingIds: List[UUID],
      aoiIds: List[UUID],
    ): Unit =
    cache.invalidateAll(projectIds ++ processingIds ++ aoiIds)

  private def get(
      ids: List[UUID],
      fetch: List[UUID] => ConnectionIO[Map[UUID, Progress]],
    ): ConnectionIO[Map[UUID, Progress]] = {
    val fromCache = cache.getAllPresent(ids)
    val toFetchFromDb = ids.filterNot(fromCache.contains)

    val fromDb =
      if (toFetchFromDb.isEmpty)
        Map[UUID, Progress]().pure[ConnectionIO]
      else
        fetch(ids)

    fromDb.map(fromCache ++ _)
  }

  private def loadProcessingProgress(ids: List[UUID]): ConnectionIO[Map[UUID, Progress]] =
    for {
      details <- ProgressRepo.getProcessingsProgressDetails(ids)
      processingProgresses <- ProcessingRepo.getProcessingProgress(ids)
      progressById = processingProgresses.map(p => p.id -> p).toMap
    } yield ids.map { id =>
      val processingProgress =
        progressById
          .get(id)
          .fold(ProjectProgress(id, Status.Unprocessed, 0, None))(p =>
            ProjectProgress(p.id, p.status, p.percentCompleted, p.estimate)
          )
      val lastStatusUpdate = lastUpdateDate(processingProgress.status, details(id))
      val progress = Progress(
        processingProgress.status,
        processingProgress.percentCompleted.toInt,
        details(id),
        lastStatusUpdate,
        processingProgress.estimate,
      )
      cache.put(id, progress)
      id -> progress
    }.toMap

  private def loadProjectProgress(projectIds: List[UUID]): ConnectionIO[Map[UUID, Progress]] =
    for {
      details <- ProgressRepo.getProjectProgressDetails(projectIds)
      projectProgresses <- ProjectRepo.getProjectProgress(projectIds)
      progressById = projectProgresses.map(p => p.id -> p).toMap
    } yield projectIds.map { id =>
      val projectProgress =
        progressById.getOrElse(id, ProjectProgress(id, Status.Unprocessed, 0, None))
      val lastStatusUpdate = lastUpdateDate(projectProgress.status, details(id))
      val progress = Progress(
        projectProgress.status,
        projectProgress.percentCompleted.toInt,
        details(id),
        lastStatusUpdate,
        projectProgress.estimate,
      )
      cache.put(id, progress)
      id -> progress
    }.toMap

  private def loadAoiProgress(ids: List[UUID]): ConnectionIO[Map[UUID, Progress]] =
    for {
      details <- ProgressRepo.getAoiProgressDetails(ids)
      processings <- ProcessingRepo.getByAoiIds(ids)
      processingById = processings.map(p => p.id -> p).toMap
      summaries <- AoiRepo.getAoiSummary(ids)
    } yield (for {
      id <- ids
      progress = Progress(details(id), summaries(id).area, processingById.get(id).map(_.created))
      _ = cache.put(id, progress)
    } yield id -> progress).toMap

  def getProjectsProgress(projects: List[ProjectDto]): ConnectionIO[Map[UUID, Progress]] =
    get(projects.map(_.id), loadProjectProgress)

  def getProcessingsProgress(processings: List[ProcessingDto]): ConnectionIO[Map[UUID, Progress]] =
    get(processings.map(_.id), loadProcessingProgress)

  def getProcessingProgress(processing: ProcessingDto): ConnectionIO[Progress] =
    get(List(processing.id), loadProcessingProgress).map(_(processing.id))

  def getAoisProgress(aois: List[AoiDto]): ConnectionIO[Map[UUID, Progress]] =
    get(aois.map(_.id), loadAoiProgress)

  def getAoiProgress(aoi: AoiDto): ConnectionIO[Progress] =
    get(List(aoi.id), loadAoiProgress).map(_(aoi.id))
}

object ProgressService {
  private val singleton = new ProgressService()
  def apply(): ProgressService = singleton
}
