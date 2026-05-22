package io.geoalert.mapflow.providers.maxar

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

import cats.syntax.either._
import com.typesafe.scalalogging.LazyLogging
import io.circe.Decoder
import io.circe.generic.semiauto.deriveDecoder

import io.geoalert.mapflow.rest.json.Decoders

import geotrellis.vector._
import geotrellis.vector.io.json.GeoJson
import geotrellis.vector.io.json.JsonFeatureCollectionMap

object SearchMetaResponseParser extends Decoders with LazyLogging {
  implicit val maxarMetaResponseDecoder: Decoder[MaxarFeatureMetadata] =
    deriveDecoder[MaxarFeatureMetadata]

  implicit val decodeInstant: Decoder[Instant] = Decoder.decodeString.emap { str =>
    Either.catchNonFatal(Instant.from(DATE_FORMAT.parse(str))).leftMap { t =>
      logger.warn("Unable to parse date", t)
      "Instant"
    }
  }

  val DATE_FORMAT: DateTimeFormatter = DateTimeFormatter
    .ofPattern("yyyy-MM-dd HH:mm:ss")
    .withZone(ZoneId.of("UTC"))

  def parseSearchMetaResponse(responseString: String): List[MaxarFeature] = {
    logger.debug(s"Maxar Catalog response: $responseString")
    val featureCollection = GeoJson.parse[JsonFeatureCollectionMap](responseString)

    val features = featureCollection.getAllPolygonFeatures[MaxarFeatureMetadata]()

    val images = for {
      (id, feature) <- features
    } yield MaxarFeature(id, feature.geom, feature.data)

    images.toList
  }
}
