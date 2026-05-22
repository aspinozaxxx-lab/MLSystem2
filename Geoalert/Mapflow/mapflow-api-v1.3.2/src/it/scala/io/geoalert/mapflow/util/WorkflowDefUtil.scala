package io.geoalert.mapflow.util

import java.util.UUID

import cats.syntax.option._
import doobie.implicits._

import io.geoalert.mapflow.model.CreateWorkflowDefInput
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.WorkflowDefRepo

object WorkflowDefUtil {
  private val yml = """
              |name: Buildings Detection With Heights
              |version: 0
              |price_per_sq_km: 11
              |stages:
              |  select-source:
              |    description: Source selection
              |    action: select-source
              |    config:
              |      params:
              |        auto_confirm: true
              |        source_type: xyz
              |        url: http://api.tiles.mapbox.com/api/tiles/{x}/y/{z}.png?token=}
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

  def createWd(name: String = "Test workflow definition", isDefault: Boolean = true): UUID = {
    val input = CreateWorkflowDefInput(None, name, None, None, Some(yml), Some(11), isDefault.some)

    WorkflowDefRepo
      .createWorkflowDef(input, 5555, "WE " + UUID.randomUUID().toString, yml)
      .transact(xa)
      .unsafeRunSync()
  }
}
