package io.geoalert.mapflow.service.importer

import java.util.UUID

/** Maps Aoi id to a tuple */
case class ImportResult(value: Map[UUID, (ImportResult.Operation, ImportResult.Area)])

object ImportResult {
  /** The Long value is the area of the AOI
    */
  type Area = Long

  /** value is an accumulator of inset-delete operations: 1 means insert, -1 means delete
    */
  type Operation = Int

  def combineAll(results: List[ImportResult]): ImportResult = {
    val map = results
      .flatMap(_.value.toList)
      .groupBy(_._1)
      .view
      .mapValues(v => (v.map(_._2._1).sum, v.head._2._2))
      .filter(_._2._1 != 0)
      .toMap

    ImportResult(map)
  }

  def apply(): ImportResult = ImportResult(Map())

  def apply(toInsert: List[(UUID, Long)], toDelete: Set[(UUID, Long)]): ImportResult = {
    val insertions = ImportResult(toInsert.map(a => (a._1, (1, a._2))).toMap)
    val deletions = ImportResult(toDelete.map(a => (a._1, (-1, a._2))).toMap)
    combineAll(List(insertions, deletions))
  }
}
