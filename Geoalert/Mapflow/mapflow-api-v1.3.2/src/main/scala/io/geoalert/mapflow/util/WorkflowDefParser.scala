package io.geoalert.mapflow.util

import scala.util.Try

import cats.implicits._
import com.typesafe.scalalogging.LazyLogging
import io.circe.generic.auto._
import io.circe.yaml.parser
import org.yaml.snakeyaml.Yaml

import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.InternalServerError
import io.geoalert.mapflow.exception.ParsingError
import io.geoalert.mapflow.model.BlockConfig
import io.geoalert.mapflow.model.DataSource
import io.geoalert.mapflow.model.DataSource.DataSource
import io.geoalert.mapflow.model.SourceType
import io.geoalert.mapflow.model.SourceType.SourceType
import io.geoalert.mapflow.model.WorkflowDefSummary

object WorkflowDefParser extends LazyLogging {
  case class BlockConfigRaw(
      name: String,
      display_name: Option[String],
      optional: Boolean,
      price: Double,
      default_enabled: Boolean,
    )

  def parseYml(yml: String): Either[ApplicationError, WorkflowDefSummary] =
    (for {
      json <- parser.parse(yml)
      params = json
        .hcursor
        .downField("stages")
        .downField("select-source")
        .downField("config")
        .downField("params")
      blockParameters <- json
        .hcursor
        .downField("blocks")
        .as[Option[Seq[BlockConfigRaw]]]
      zoomField <- params.downField("zoom").as[Option[Int]]
      zoom <- Right(zoomField)
      sourceTypeField <- params.downField("source_type").as[Option[String]]
      sourceTypeOpt <- Right(sourceTypeField.flatMap(SourceType.find))
      urlField = params.downField("url").as[Option[String]]
      urlOpt = urlField.toOption.flatten
      sourceOpt = for {
        sourceType <- sourceTypeOpt
        url <- urlOpt
      } yield extractDataSourceFromUrl(sourceType, url)
      source <- Right(sourceOpt)
      nameOpt <- json.hcursor.get[String]("name")
      name <- Right(nameOpt)
      pricePerSqKmField <- json.hcursor.downField("price_per_sq_km").as[Option[Double]]
      partitionSize <- json.hcursor.downField("partition_size").as[Option[Double]]
      pricePerSqKm <- Right(pricePerSqKmField.getOrElse(0.0))
      userInputParams = json
        .hcursor
        .downField("stages")
        .downField("user-input")
        .downField("config")
        .downField("params")
      userInputBucket <- userInputParams.downField("bucket").as[Option[String]]
      blocks = blockParameters.map(
        _.map(bp =>
          BlockConfig(
            bp.name,
            bp.display_name.getOrElse(bp.name),
            bp.optional,
            bp.price,
            bp.default_enabled,
          )
        )
      )
    } yield WorkflowDefSummary(
      name,
      sourceTypeOpt,
      source,
      pricePerSqKm,
      zoom,
      urlOpt,
      userInputBucket,
      partitionSize,
      blocks.getOrElse(Seq()),
    )).left.map { ex =>
      logger.error("Cannot parse YML", ex)
      InternalServerError(s"Cannot parse yml $yml")
    }

  def extractDataSourceFromUrl(sourceType: SourceType, urlString: String): DataSource =
    sourceType match {
      case SourceType.xyz => DataSource.tiles
      case SourceType.tms => DataSource.tiles
      case SourceType.quadkey => DataSource.tiles
      case SourceType.sentinel_l2a => DataSource.sentinel_l2a
      case SourceType.local => DataSource.local
      case _ =>
        throw new IllegalStateException(
          s"Unexpected source_type/url combination $sourceType, $urlString"
        )
    }

  def extractDataSource(sourceType: SourceType): DataSource =
    sourceType match {
      case SourceType.xyz => DataSource.tiles
      case SourceType.tms => DataSource.tiles
      case SourceType.quadkey => DataSource.tiles
      case SourceType.sentinel_l2a => DataSource.sentinel_l2a
      case SourceType.local => DataSource.local
      case _ =>
        throw new IllegalStateException(
          s"Unexpected source_type $sourceType"
        )
    }

  def updateYml(
      yml: String,
      name: String,
      version: Int,
    ): Either[ApplicationError, String] = {
    import scala.jdk.CollectionConverters._

    val yaml = new Yaml()

    for {
      map <- parseYmlToMap(yml)
      updatedMap = map
        .updated("name", name)
        .updated("version", version)
    } yield yaml.dump(updatedMap.asJava)
  }

  private def parseYmlToMap(definition: String): Either[ParsingError, Map[String, AnyRef]] = {
    import scala.jdk.CollectionConverters._
    val yaml = new Yaml()

    Try(
      yaml
        .load[java.util.Map[String, AnyRef]](definition)
        .asScala
        .toMap
    ).toEither
      .leftMap(ParsingError(_))
  }
}
