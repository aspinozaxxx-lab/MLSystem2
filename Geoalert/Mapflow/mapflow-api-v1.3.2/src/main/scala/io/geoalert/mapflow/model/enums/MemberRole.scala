package io.geoalert.mapflow.model.enums

import enumeratum.EnumEntry.Snakecase
import enumeratum._

sealed trait MemberRole extends Snakecase
object MemberRole extends Enum[MemberRole] with CirceEnum[MemberRole] {
  case object Owner extends MemberRole
  case object Readonly extends MemberRole
  case object Maintainer extends MemberRole
  case object Contributor extends MemberRole
  override def values: IndexedSeq[MemberRole] = findValues
}
