package io.geoalert.mapflow.repo

import java.util.UUID
import cats.data.EitherT
import cats.data.NonEmptyList
import cats.syntax.applicative._
import cats.syntax.option._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.model._

object RasterLayerRepo
    extends GenericRepo[RasterLayer](
      "raster_layer",
      Seq(
        "id",
        "uri",
      ),
    ) {
  def getRasterLayers(ids: Option[Seq[UUID]]): ConnectionIO[List[RasterLayer]] = ids match {
    case Some(ids) => getAllByIds(ids)
    case None => getAll()
  }

  def getRasterLayer(id: UUID): EitherT[ConnectionIO, NotFound, RasterLayer] =
    EitherT.fromOptionF(getOneById(id), NotFound[RasterLayer](id))

  def getAllByAoiIds(aoiIds: List[UUID]): ConnectionIO[Map[UUID, RasterLayer]] = {
    def batch(aoiIds: List[UUID]) = NonEmptyList.fromList(aoiIds) match {
      case Some(aoiIds) =>
        val sql =
          const(s"SELECT a.id, p.raster_layer_id FROM $dbSchema.processing p JOIN $dbSchema.aoi a ON a.processing_id = p.id WHERE") ++
            in(fr"a.id", aoiIds)
        for {
          rlIdsByAoiIds <- sql.query[(UUID, UUID)].to[List].map(_.toMap)
          rls <- getRasterLayers(rlIdsByAoiIds.values.toList.distinct.some)
          rlsByIds = rls.map(rl => (rl.id, rl)).toMap
        } yield rlIdsByAoiIds.view.mapValues(rlsByIds).toList
      case None => List[(UUID, RasterLayer)]().pure[ConnectionIO]
    }
    aoiIds.batchTraverse(maxInFrLen)(batch).map(_.toMap)
  }

  def createRasterLayer(uri: String): ConnectionIO[RasterLayer] = {
    val fields = Seq(
      Some("uri" -> uri)
    ).flatten.toMap

    for {
      id <- create(fields)
      maybeRasterLayer <- getRasterLayer(id).rethrowT
    } yield maybeRasterLayer
  }
}
