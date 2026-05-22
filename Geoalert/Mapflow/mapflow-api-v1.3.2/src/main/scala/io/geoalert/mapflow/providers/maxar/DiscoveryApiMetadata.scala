package io.geoalert.mapflow.providers.maxar

case class DiscoveryApiMetadata(
    sunElevationAvg: Double,
    sunAzimuthAvg: Double,
    targetAzimuthAvg: Double,
    offNadirAvg: Double,
    vehicleName: String,
  )
