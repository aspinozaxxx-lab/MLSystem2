package io.geoalert.mapflow.model

object DataSource extends Enumeration {
  type DataSource = Value
  protected case class Val(pricePerMp: Double) extends super.Val

  implicit def valueToDataSourceVal(x: Value): Val = x.asInstanceOf[Val]

  val local: Val = Val(0)
  val tiles: Val = Val(0)
  val sentinel_l2a: Val = Val(0.09) /* Cost per granule here */
}
