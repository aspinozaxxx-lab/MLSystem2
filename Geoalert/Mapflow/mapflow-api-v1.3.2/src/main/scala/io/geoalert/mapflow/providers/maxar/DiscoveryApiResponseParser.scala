package io.geoalert.mapflow.providers.maxar

import com.typesafe.scalalogging.LazyLogging
import io.circe._
import io.circe.generic.auto._

object DiscoveryApiResponseParser extends LazyLogging {
  private case class DiscoveryApiResponse(features: Seq[DiscoveryApiFeature])
  private case class DiscoveryApiFeature(attributes: Map[String, Any])

  implicit val objDecoder: Decoder[Any] = {
    case x: HCursor if x.value.isString =>
      x.value.as[String]
    case x: HCursor if x.value.isBoolean =>
      x.value.as[Boolean]
    case x: HCursor if x.value.isNumber =>
      x.value.as[Double]
    case x: HCursor if x.value.isNull =>
      x.value.as[Unit]
    case _ => throw new RuntimeException("Unexpected field type")
  }

  def parseResponse(responseString: String): Option[DiscoveryApiMetadata] = {
    logger.debug(s"Maxar Discovery API response: $responseString")

    parser.decode[DiscoveryApiResponse](responseString) match {
      case Left(err) =>
        logger.error("Unable to parse Discovery API response", err)
        None
      case Right(response) =>
        for {
          feature <- response.features.headOption
          sunElevationAvg <- feature.attributes.get("sun_elevation_avg").map(_.toString.toDouble)
          sunAzimuthAvg <- feature.attributes.get("sun_azimuth_avg").map(_.toString.toDouble)
          targetAzimuthAvg <- feature.attributes.get("target_azimuth_avg").map(_.toString.toDouble)
          offNadirAvg <- feature.attributes.get("off_nadir_avg").map(_.toString.toDouble)
          vehicleName <- feature.attributes.get("vehicle_name").map(_.toString)
        } yield DiscoveryApiMetadata(
          sunElevationAvg,
          sunAzimuthAvg,
          targetAzimuthAvg,
          offNadirAvg,
          vehicleName,
        )
    }
  }
}
