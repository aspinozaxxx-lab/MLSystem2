package io.geoalert.mapflow.model

object SourceType extends Enumeration {
  type SourceType = Value

  val xyz, tms, quadkey, sentinel_l2a, local = Value

  def find(name: String): Option[SourceType] =
    values.find(_.toString.equals(name))
}
