package io.geoalert.mapflow.service.importer

import java.util.UUID

import cats.Monoid

import geotrellis.vector._

case class MergeResult(toInsert: List[Projected[Geometry]], toDelete: Set[(UUID, Long)]) {
  def filter(f: Projected[Geometry] => Boolean) = MergeResult(toInsert filter f, toDelete)
}

object MergeResult {
  implicit val mergeResultMonoid: Monoid[MergeResult] = new Monoid[MergeResult] {
    override def empty: MergeResult = MergeResult()

    override def combine(x: MergeResult, y: MergeResult): MergeResult =
      MergeResult(x.toInsert ++ y.toInsert, x.toDelete ++ y.toDelete)
  }

  def apply(): MergeResult = MergeResult(List(), Set())
}
