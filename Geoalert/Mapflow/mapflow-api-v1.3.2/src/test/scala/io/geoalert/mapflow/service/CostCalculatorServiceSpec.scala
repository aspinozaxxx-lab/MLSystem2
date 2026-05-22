package io.geoalert.mapflow.service

import cats.implicits.catsSyntaxOptionId
import io.geoalert.mapflow.model.{BlockConfig, DataProvider, DataSource, ProcessingParams, SourceType, WorkflowDefSummary}
import io.geoalert.mapflow.repo.BlockParameters
import org.scalatest.{FunSpec, Matchers}

import java.util.UUID

class CostCalculatorServiceSpec extends FunSpec with Matchers with Services {
  describe("CostCalculatorService") {
    it("Should calculate meters per px") {
      val mpp = costCalculatorService.metersPerPxEst(18)
      mpp should be(0.3425193530387684 )
    }

    it("Should calculate MP per sq. km") {
      val mpp = costCalculatorService.megapixelsInSqKm(18)
      mpp should be(8.523731677829918)
    }

    it("Should calculate cost for data provider") {
      val cost = costCalculatorService.estimateCost(
        WorkflowDefSummary("", Some(SourceType.xyz), Some(DataSource.tiles), 0, Some(18), Some("http://example.com/{z}/{x}/{y}.png"), None, None, Seq()),
        ProcessingParams(Map()),
        3.5, 
        Some(DataProvider(UUID.fromString("61cd6899-19e8-44a0-97db-b86f1a9b7af4"), "", "", None, None, 1, None, None, None, isDefault = false, None)),
        Seq(),
      )
      cost should be(30)
    }

    it("Should calculate cost for processing only when data source is xyz") {
      val cost = costCalculatorService.estimateCost(
        WorkflowDefSummary("", Some(SourceType.xyz), Some(DataSource.tiles), 10, Some(18), Some("http://example.com/{z}/{x}/{y}.png"), None, None, Seq()),
        ProcessingParams(Map()), 3.5, None, Seq())
      cost should be(35)
    }    
    
    it("Should calculate cost for processing only when data source is local") {
      val cost = costCalculatorService.estimateCost(
        WorkflowDefSummary("", Some(SourceType.local), Some(DataSource.local), 10, Some(18), Some("s3://users-data/username/input.tif"), None, None, Seq()),
        ProcessingParams(Map()), 3.5, None, Seq())
      cost should be(35)
    }

    it("Should calculate cost for both data provider and processing") {
      val cost = costCalculatorService.estimateCost(
        WorkflowDefSummary("", Some(SourceType.xyz), Some(DataSource.tiles), 10, Some(18), Some("http://example.com/{z}/{x}/{y}.png"), None, None, Seq()),
        ProcessingParams(Map()), 
        3.5, 
        Some(DataProvider(UUID.fromString("61cd6899-19e8-44a0-97db-b86f1a9b7af4"), "", "", None, None, 1, None, None, None, isDefault = false, None)),
        Seq()
      )
      cost should be(65)
    }

    it("Should calculate cost based on blocks using block defaults") {
      val cost = costCalculatorService.estimateCost(
        WorkflowDefSummary("",
          Some(SourceType.xyz),
          Some(DataSource.tiles),
          10,
          Some(18),
          Some("http://example.com/{z}/{x}/{y}.png"),
          None,
          None,
          Seq(BlockConfig("inference", "Segmentation", optional = false, 5, defaultEnabled = true), BlockConfig("simplification", "Simplification", optional = true, 7, defaultEnabled = true)),
        ),
        ProcessingParams(Map()),
        10,
        Some(DataProvider(UUID.fromString("61cd6899-19e8-44a0-97db-b86f1a9b7af4"), "", "", None, None, 0, None, None, None, isDefault = false, None)),
        Seq()
      )
      cost should be(120)
    }

    it("Should calculate cost based on blocks") {
      val cost = costCalculatorService.estimateCost(
        WorkflowDefSummary("",
          Some(SourceType.xyz),
          Some(DataSource.tiles),
          10,
          Some(18),
          Some("http://example.com/{z}/{x}/{y}.png"),
          None,
          None,
          Seq(BlockConfig("inference", "Segmentation", optional = false, 5, defaultEnabled = true), BlockConfig("simplification", "Simplification", optional = true, 7, defaultEnabled = true)),
        ),
        ProcessingParams(Map()),
        10,
        Some(DataProvider(UUID.fromString("61cd6899-19e8-44a0-97db-b86f1a9b7af4"), "", "", None, None, 0, None, None, None, isDefault = false, None)),
        Seq(BlockParameters("simplification", enabled = true, "Segmentation".some))
      )
      cost should be(120)
    }

    it("Should round area up to 1 sq. km") {
      val cost = costCalculatorService.estimateCost(
        WorkflowDefSummary("", Some(SourceType.xyz), Some(DataSource.tiles), 10, Some(18), Some("http://example.com/{z}/{x}/{y}.png"), None, None, Seq()),
        ProcessingParams(Map()), 
        0.1, 
        Some(DataProvider(UUID.fromString("61cd6899-19e8-44a0-97db-b86f1a9b7af4"), "", "", None, None, 1, None, None, None, isDefault = false, None)),
        Seq(),
      )
      cost should be(19)
    }

  }
}
