package io.geoalert.mapflow.util

import io.geoalert.mapflow.model.{BlockConfig, DataSource, SourceType, WorkflowDefSummary}
import org.scalatest.{FunSpec, Matchers}

class WorkflowDefParserSpec extends FunSpec with Matchers {
  describe("YML parser") {
    it("should parse complete YML") {
      val yml =
        """
          |name: Buildings Detection With Heights
          |version: 0
          |price_per_sq_km: 5
          |partition_size: 0.01
          |stages:
          |  select-source:
          |    description: Source selection
          |    action: select-source
          |    config:
          |      params:
          |        auto_confirm: true
          |        source_type: xyz
          |        url: http://10.1.6.1:8800/api/tiles?proxy=true&url=http://mt0.google.com/vt/lyrs=s%26hl=en%26x={x}%26y={y}%26z={z}%26s=Ga
          |        zoom: 18
          |        projection: epsg:3857
          |blocks:
          |  - name: inference
          |    display_name: Segmentation
          |    optional: false
          |    price: 5
          |    default_enabled: true
          |  - name: simplification
          |    display_name: Simplification
          |    optional: true
          |    price: 7
          |    default_enabled: false
          |""".stripMargin

      val wd = WorkflowDefParser.parseYml(yml)
      wd should matchPattern {
        case Right(WorkflowDefSummary(
          "Buildings Detection With Heights",
          Some(SourceType.xyz),
          Some(DataSource.tiles),
          5,
          Some(18),
          Some("http://10.1.6.1:8800/api/tiles?proxy=true&url=http://mt0.google.com/vt/lyrs=s%26hl=en%26x={x}%26y={y}%26z={z}%26s=Ga"),
          None,
          Some(0.01),
          Seq(BlockConfig("inference", "Segmentation", false, 5.0, true), BlockConfig("simplification", "Simplification", true, 7.0, false))
        )) =>
      }
    }

    it("should parse complete YML with URL without parameters") {
      val yml =
        """
          |name: Buildings Detection With Heights
          |version: 0
          |stages:
          |  select-source:
          |    description: Source selection
          |    action: select-source
          |    config:
          |      params:
          |        auto_confirm: true
          |        source_type: xyz
          |        url: http://example.com
          |        zoom: 18
          |        projection: epsg:3857
          |""".stripMargin

      val wd = WorkflowDefParser.parseYml(yml)
      wd should matchPattern {
        case Right(WorkflowDefSummary("Buildings Detection With Heights", Some(SourceType.xyz), Some(DataSource.tiles), 0, Some(18), Some("http://example.com"), None, None, Seq())) =>
      }

    }

    it("should not parse YML without stages") {
      val yml =
        """
          |name: Buildings Detection With Heights
          |version: 0
          |""".stripMargin

      val wd = WorkflowDefParser.parseYml(yml)
      wd should matchPattern {
        case Right(WorkflowDefSummary("Buildings Detection With Heights", None, None, 0, None, None, None, None, Seq())) =>
      }
    }

    it("should parse YML with unknown source type") {
      val yml =
        """
          |name: Buildings Detection With Heights
          |version: 0
          |stages:
          |  select-source:
          |    description: Source selection
          |    action: select-source
          |    config:
          |      params:
          |        auto_confirm: true
          |        source_type: wms
          |        url: http://example.com
          |        zoom: 18
          |        projection: epsg:3857
          |""".stripMargin

      val wd = WorkflowDefParser.parseYml(yml)
      wd should matchPattern {
        case Right(WorkflowDefSummary("Buildings Detection With Heights", None, None, 0, Some(18), Some("http://example.com"), None, None, Seq())) =>
      }
    }

    it("should parse YML with invalid URL") {
      val yml =
        """
          |name: Buildings Detection With Heights
          |version: 0
          |stages:
          |  select-source:
          |    description: Source selection
          |    action: select-source
          |    config:
          |      params:
          |        auto_confirm: true
          |        source_type: sentinel_l2a
          |        url: L1C_T37UEA_A026187_20200627T082607
          |        zoom: 18
          |        projection: epsg:3857
          |""".stripMargin

      val wd = WorkflowDefParser.parseYml(yml)
      wd should matchPattern {
        case Right(WorkflowDefSummary("Buildings Detection With Heights", Some(SourceType.sentinel_l2a), Some(DataSource.sentinel_l2a), 0, Some(18), Some("L1C_T37UEA_A026187_20200627T082607"), None, None, Seq())) =>
      }
    }
    it("should parse YML with user-input") {
      val yml =
        """
          |name: Buildings Detection With Heights
          |version: 0
          |stages:
          |  user-input:
          |    description: Waiting for user input
          |    action: user-input
          |    dependsOn:
          |    - load-data
          |    config:
          |      params:
          |        bucket: workflow-white-maps
          |        inputs: meta.geojson,shadows_labels.geojson,walls_labels.geojson
          |        recipients: dev@geoalert.io
          |""".stripMargin

      val wd = WorkflowDefParser.parseYml(yml)
      wd should matchPattern {
        case Right(WorkflowDefSummary("Buildings Detection With Heights", None, None, 0, None, None, Some("workflow-white-maps"), None, Seq())) =>
      }
    }

    it("Should update WD name and version") {
      val yml =
        """
          |stages:
          |  user-input:
          |    description: Waiting for user input
          |    action: user-input
          |    dependsOn:
          |    - load-data
          |    config:
          |      params:
          |        bucket: workflow-white-maps
          |        inputs: meta.geojson,shadows_labels.geojson,walls_labels.geojson
          |        recipients: dev@geoalert.io
          |""".stripMargin


      val wd = for {
        updatedYaml <- WorkflowDefParser.updateYml(yml, "new name", 42)
        summary <- WorkflowDefParser.parseYml(updatedYaml)
      } yield summary

      wd should matchPattern {
        case Right(WorkflowDefSummary("new name", _, _, _, _, _, _, _, _)) =>
      }
    }
  }
}
