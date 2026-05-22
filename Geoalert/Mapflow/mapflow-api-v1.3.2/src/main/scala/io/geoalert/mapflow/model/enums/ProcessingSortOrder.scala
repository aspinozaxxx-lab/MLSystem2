package io.geoalert.mapflow.model.enums

import enumeratum.EnumEntry.UpperSnakecase
import enumeratum._

sealed trait ProcessingSortOrder extends UpperSnakecase
object ProcessingSortOrder extends Enum[ProcessingSortOrder] with CirceEnum[ProcessingSortOrder] {
  case object Name extends ProcessingSortOrder
  case object Scenario extends ProcessingSortOrder
  case object Created extends ProcessingSortOrder
  case object Completed extends ProcessingSortOrder
  case object ProjectName extends ProcessingSortOrder
  case object Email extends ProcessingSortOrder
  case object Status extends ProcessingSortOrder
  case object Progress extends ProcessingSortOrder
  override def values: IndexedSeq[ProcessingSortOrder] = findValues
}
