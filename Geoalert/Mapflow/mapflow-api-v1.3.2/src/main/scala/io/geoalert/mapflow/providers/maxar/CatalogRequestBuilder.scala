package io.geoalert.mapflow.providers.maxar

import java.time.ZoneId
import java.time.format.DateTimeFormatter

import io.geoalert.mapflow.implicits.GeometryOps.ProjectedGeometryOps

import geotrellis.proj4.LatLng
import geotrellis.vector.Geometry
import geotrellis.vector.Projected

object CatalogRequestBuilder {
  val REQUEST_DATE_FORMAT: DateTimeFormatter = DateTimeFormatter
    .ofPattern("yyyy-MM-dd'T'hh:mm:ssxxx")
    .withZone(ZoneId.of("UTC"))

  def buildRequest(input: MaxarCatalogRequest): String = {
    val dateFilter: String = buildRangeFilter(
      "acquisitionDate",
      input.acquisitionDateFrom.map(REQUEST_DATE_FORMAT.format(_)),
      input.acquisitionDateTo.map(REQUEST_DATE_FORMAT.format(_)),
    )

    val maxCloudFilter: String = buildMaxCloudFilter(input.maxCloudCover)

    val resolutionFilter = buildRangeFilter(
      "groundSampleDistance",
      input.minResolution.map(_.toString),
      input.maxResolution.map(_.toString),
    )

    val offNadirAngleFilter = buildRangeFilter(
      "offNadirAngle",
      input.minOffNadirAngle.map(_.toString),
      input.maxOffNadirAngle.map(_.toString),
    )
    val geometryFilter: String = buildGeometryFilter(input.aoi)

    val featureIdFilter = buildFeatureIdFilter(input.featureId)

    s"""<?xml version="1.0" encoding="utf-8"?>
       |<GetFeature
       |  service="wfs"
       |  version="1.1.0"
       |  outputFormat="json"
       |  xmlns="http://www.opengis.net/wfs"
       |  xmlns:ogc="http://www.opengis.net/ogc">
       |  <Query typeName="DigitalGlobe:FinishedFeature" srsName="EPSG:4326">
       |    <PropertyName>productType</PropertyName>
       |    <PropertyName>source</PropertyName>
       |    <PropertyName>colorBandOrder</PropertyName>
       |    <PropertyName>cloudCover</PropertyName>
       |    <PropertyName>offNadirAngle</PropertyName>
       |    <PropertyName>acquisitionDate</PropertyName>
       |    <PropertyName>geometry</PropertyName>
       |    <PropertyName>groundSampleDistance</PropertyName>
       |    <PropertyName>legacyId</PropertyName>
       |    <ogc:Filter>
       |      <ogc:And>
       |        $dateFilter
       |        $maxCloudFilter
       |        $geometryFilter
       |        $resolutionFilter
       |        $featureIdFilter
       |        $offNadirAngleFilter
       |      </ogc:And>
       |    </ogc:Filter>
       |    <ogc:SortBy>
       |      <ogc:SortProperty>
       |        <ogc:PropertyName>acquisitionDate</ogc:PropertyName>
       |        <ogc:SortOrder>DESC</ogc:SortOrder>
       |      </ogc:SortProperty>
       |    </ogc:SortBy>
       |  </Query>
       |</GetFeature>
       |""".stripMargin
  }

  private def buildFeatureIdFilter(valueOpt: Option[String]): String =
    valueOpt match {
      case Some(value) =>
        s"""
           |<ogc:PropertyIsEqualTo>
           |  <ogc:PropertyName>featureId</ogc:PropertyName>
           |  <ogc:Literal>$value</ogc:Literal>
           |</ogc:PropertyIsEqualTo>
           |""".stripMargin
      case None => ""
    }

  private def buildGeometryFilter(aoi: Option[Geometry]): String =
    aoi match {
      case Some(geometry) =>
        val gml = Projected(geometry, LatLng.epsgCode.get).toGml3

        s"""<ogc:Intersects>
           |  <ogc:PropertyName>geometry</ogc:PropertyName>
           |  $gml
           |</ogc:Intersects>
           |""".stripMargin
      case None => ""
    }

  private def propertyRangeFilter(
      propertyName: String,
      lower: String,
      upper: String,
    ): String =
    s"""
       |<ogc:PropertyIsBetween>
       |  <ogc:PropertyName>$propertyName</ogc:PropertyName>
       |  <ogc:LowerBoundary>
       |    <ogc:Literal>$lower</ogc:Literal>
       |  </ogc:LowerBoundary>
       |  <ogc:UpperBoundary>
       |    <ogc:Literal>$upper</ogc:Literal>
       |  </ogc:UpperBoundary>
       |</ogc:PropertyIsBetween>""".stripMargin

  private def propertyGreaterFilter(propertyName: String, lower: String): String =
    s"""
      |<ogc:Or>
      | <ogc:PropertyIsGreaterThanOrEqualTo>
      |   <ogc:PropertyName>$propertyName</ogc:PropertyName>
      |              <ogc:Literal>$lower</ogc:Literal>
      | </ogc:PropertyIsGreaterThanOrEqualTo>
      | <ogc:PropertyIsNull>
      |   <ogc:PropertyName>$propertyName</ogc:PropertyName>
      | </ogc:PropertyIsNull>
      |</ogc:Or>
      |      |""".stripMargin

  private def propertyLowerFilter(propertyName: String, upper: String): String =
    s"""
       |<ogc:Or>
       | <ogc:PropertyIsLessThanOrEqualTo>
       |   <ogc:PropertyName>$propertyName</ogc:PropertyName>
       |              <ogc:Literal>$upper</ogc:Literal>
       | </ogc:PropertyIsLessThanOrEqualTo>
       | <ogc:PropertyIsNull>
       |   <ogc:PropertyName>$propertyName</ogc:PropertyName>
       | </ogc:PropertyIsNull>
       |</ogc:Or>
       |       |""".stripMargin

  private def buildRangeFilter(
      propertyName: String,
      dateFrom: Option[String],
      dateTo: Option[String],
    ): String =
    (dateFrom, dateTo) match {
      case (None, None) => ""
      case (Some(fromValue), Some(toValue)) => propertyRangeFilter(propertyName, fromValue, toValue)
      case (Some(fromValue), None) => propertyGreaterFilter(propertyName, fromValue)
      case (None, Some(toValue)) => propertyLowerFilter(propertyName, toValue)
    }

  private def buildMaxCloudFilter(maxCloudCover: Option[Double]): String = {
    val maxCloudFilter = maxCloudCover match {
      case Some(value) => propertyLowerFilter("cloudCover", s"$value")
      case None => ""
    }
    maxCloudFilter
  }
}
