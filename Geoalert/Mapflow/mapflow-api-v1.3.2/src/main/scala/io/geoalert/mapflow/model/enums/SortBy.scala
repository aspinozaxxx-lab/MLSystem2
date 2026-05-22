package io.geoalert.mapflow.model.enums

import enumeratum.EnumEntry.Uppercase
import enumeratum._

sealed trait SortBy extends Uppercase
object SortBy extends Enum[SortBy] with CirceEnum[SortBy] {
  case object ASC extends SortBy
  case object DESC extends SortBy
  override def values: IndexedSeq[SortBy] = findValues
}
