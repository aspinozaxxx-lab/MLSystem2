package io.geoalert.mapflow.repo

import java.util.UUID

import cats.data.EitherT
import doobie.ConnectionIO
import doobie.implicits._
import doobie.postgres.implicits._

import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model._

object VectorLayerRepo
    extends GenericRepo[VectorLayer](
      "vector_layer",
      Seq(
        "id",
        "external_id",
        "name",
      ),
    ) {
  def getVectorLayers(ids: Option[Seq[UUID]]): ConnectionIO[List[VectorLayer]] = ids match {
    case Some(ids) => getAllByIds(ids)
    case None => getAll()
  }

  def getVectorLayer(id: UUID): EitherT[ConnectionIO, NotFound, VectorLayer] =
    EitherT.fromOptionF(getOneById(id), NotFound[VectorLayer](id))

  def createVectorLayer(name: String, externalId: UUID): ConnectionIO[VectorLayer] = {
    val fields = Seq(
      Some("name" -> name),
      Some("external_id" -> externalId),
    ).flatten.toMap

    for {
      id <- create(fields)
      vectorLayer <- getVectorLayer(id).rethrowT
    } yield vectorLayer
  }
}
