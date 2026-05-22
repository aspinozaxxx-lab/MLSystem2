package io.geoalert.mapflow.model

import scala.util.Try

import cats.data.EitherT
import cats.syntax.either._
import doobie._
import doobie.implicits._

import io.geoalert.mapflow.exception.GeometryError
import io.geoalert.mapflow.implicits.GeometryOps._
import io.geoalert.mapflow.service.importer.MergeResult

import geotrellis.vector._

sealed abstract class MergeStrategy(val repr: String) {
  type Result = EitherT[ConnectionIO, GeometryError, MergeResult]

  override def toString = repr

  def merge(p: Projected[Geometry], old: => ConnectionIO[List[Aoi]]): Result
}

object MergeStrategy {

  /** Adds new aois without even looking if old ones exist */
  case object IgnoreExisting extends MergeStrategy("IGNORE_EXISTING") {
    override def merge(p: Projected[Geometry], old: => ConnectionIO[List[Aoi]]): Result =
      EitherT.rightT[ConnectionIO, GeometryError](MergeResult(List(p), Set()))
  }

  /** Subtracts old aois from new ones, inserts diffs only.
    * May result in overlapping aois.
    * No old aois are ever deleted.
    */
  case object InsertDiffs extends MergeStrategy("INSERT_DIFFS") {
    override def merge(p: Projected[Geometry], old: => ConnectionIO[List[Aoi]]): Result = for {
      old <- EitherT.right[GeometryError](old)
      toInsert <- safeDifference(p, old)
    } yield MergeResult(List(toInsert), Set())
  }

  /** Replaces old aois with new ones.
    * For aois that are already in progress uses `InsertDiffs` strategy.
    */
  case object ReplaceExisting extends MergeStrategy("REPLACE_EXISTING") {
    override def merge(p: Projected[Geometry], old: => ConnectionIO[List[Aoi]]): Result = for {
      old <- EitherT.right[GeometryError](old)
      (deletable, nonDeletable) = old.span(a => a.progress.status == Status.Unprocessed)
      toInsert <- safeDifference(p, nonDeletable)
      toDelete = deletable.map(a => (a.id, a.area)).toSet
    } yield MergeResult(List(toInsert), toDelete)
  }

  /** Replaces old aois with the union of old and new ones.
    * For aois that are already in progress uses `InsertDiffs` strategy.
    */
  case object Union extends MergeStrategy("UNION") {
    override def merge(p: Projected[Geometry], old: => ConnectionIO[List[Aoi]]): Result = for {
      old <- EitherT.right[GeometryError](old)
      (deletable, nonDeletable) = old.span(a => a.progress.status == Status.Unprocessed)
      diff <- safeDifference(p, nonDeletable)
      toInsert <- safeUnion(diff, deletable)
      toDelete = deletable.map(a => (a.id, a.area)).toSet
    } yield MergeResult(List(toInsert), toDelete)
  }

  private def safeDifference(g: Projected[Geometry], aois: List[Aoi]) = {
    val geomOrErr = Try(aois.foldLeft(g)((acc, a) => acc.differenceSafer(a.geometry))).toEither
    EitherT.fromEither[ConnectionIO](
      geomOrErr.leftMap(e => GeometryError(s"Error merging $aois with $g: $e"))
    )
  }

  private def safeUnion(g: Projected[Geometry], aois: List[Aoi]) = {
    val geomOrErr = Try(aois.foldLeft(g)((acc, a) => acc.union(a.geometry))).toEither
    EitherT.fromEither[ConnectionIO](
      geomOrErr.leftMap(e => GeometryError(s"Error merging $aois with $g: $e"))
    )
  }

  def apply(repr: String) = repr match {
    case InsertDiffs.repr => InsertDiffs
    case _ => sys.error(s"Invalid MergeStrategy: $repr")
  }
}
