package io.geoalert.mapflow.providers.maxar

import com.typesafe.scalalogging.LazyLogging

object MaxarSatelliteOrbitHeight extends LazyLogging {
  def orbitMeanHeightMeters(vehicle: String): Double =
    vehicle match {
      case "WV01" => 500_000
      case "WV02" => 772_000
      case "WV03" => 620_000
      case "WV04" => 610_000
      case "GE01" => 680_000
      case unknown =>
        logger.warn(
          s"Unexpected satellite platform (vehicle_name) $unknown. Setting default value 600km"
        )
        600_000
    }
}
