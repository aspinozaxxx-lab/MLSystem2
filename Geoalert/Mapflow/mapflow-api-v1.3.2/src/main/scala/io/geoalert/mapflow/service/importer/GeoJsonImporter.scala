package io.geoalert.mapflow.service.importer

import java.util.UUID

import scala.util.Try

import cats.data.EitherT
import cats.instances.list._
import cats.syntax.applicative._
import cats.syntax.either._
import cats.syntax.foldable._
import cats.syntax.option._
import cats.syntax.traverse._
import doobie._
import doobie.implicits._

import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.implicits.GeometryOps._
import io.geoalert.mapflow.model.AoiFilter
import io.geoalert.mapflow.model.MergeStrategy
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.service.AoiService

import geotrellis.proj4.LatLng
import geotrellis.vector._

class GeoJsonImporter(aoiService: AoiService) {
  type PP = Projected[Geometry]

  def importPolygon(
      p: PP,
      processingId: UUID,
      mergeStrategy: MergeStrategy,
    )(
      user: User
    ): EitherT[ConnectionIO, AoiImportError, ImportResult] = {

    // TODO: Since we don't need to change AOIs on flight, we can preprocess data without any DB requests
    // TODO stream old polygons for the case when a large area is imported
    val filter = AoiFilter(processingIds = List(processingId).some, geometry = p.some)
    lazy val old = aoiService
      .getAois(filter, None, None, None)(user)
      .map(_.aois)

    val simplified = Try(p.buffer0AndSimplify(aoiSmpl).toPolygons)
      .toEither
      .leftMap(GeometryError(_): AoiImportError)
      .pure[ConnectionIO]

    def validateProjection(
        geometries: List[Projected[Geometry]]
      ): EitherT[ConnectionIO, AoiImportError, List[Projected[Geometry]]] = {
      val valid = geometries
        .map { p =>
          val extent = p.geom.extent
          p.srid == LatLng.epsgCode.get &&
          extent.xmin >= -360 &&
          extent.xmax <= 360 &&
          extent.ymin >= -90 &&
          extent.ymax <= 90
        }
        .reduce(_ && _)

      EitherT.cond(
        valid,
        geometries,
        new GeometryError(
          s"Polygon extent $geometries is expected to be in range from [-180, -90] to [180, 90]"
        ),
      )
    }

    def merge(ps: List[PP]): EitherT[doobie.ConnectionIO, GeometryError, MergeResult] =
      ps.map(mergeStrategy.merge(_, old))
        .sequence
        .map(_.combineAll)
        .map { r =>
          val polygons = r
            .toInsert
            .map(_.buffer0)
            .toPolygons
            .map(_.geom)

          val geom =
            if (polygons.size > 1)
              List(MultiPolygon(polygons).withSRID(p.srid))
            else if (polygons.size == 1)
              List(polygons.head.withSRID(p.srid))
            else
              List()

          MergeResult(geom, r.toDelete)
        }

    // Filters out small geometries, returns Left if none are left.
    def rejectTooSmall(ps: List[PP]) = {
      val filtered = ps.filter(_.getArea >= zeroArea)
      EitherT.cond[ConnectionIO](filtered.nonEmpty, filtered, TooSmallGeometry: AoiImportError)
    }

    def createAoi(p: PP) =
      AoiRepo.createAoi(processingId, p)

    for {
      simplified <- EitherT(simplified)
      simplified <- validateProjection(simplified)
      ps <- rejectTooSmall(simplified)
      mergeRes <- merge(ps)
      toInsert <- rejectTooSmall(mergeRes.toInsert)
      ids <- EitherT.right(toInsert.traverse(createAoi))
      _ <- EitherT.right(AoiRepo.deleteByIds(mergeRes.toDelete.toList.map(_._1)))
    } yield ImportResult(ids, mergeRes.toDelete)
  }
}
