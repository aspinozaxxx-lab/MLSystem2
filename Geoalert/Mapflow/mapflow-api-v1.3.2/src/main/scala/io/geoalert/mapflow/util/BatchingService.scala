package io.geoalert.mapflow.util

import scala.annotation.tailrec
import scala.collection.parallel.CollectionConverters._

import io.geoalert.mapflow.Config.defaultPartitionSize
import io.geoalert.mapflow.Config.maxPartitionSize
import io.geoalert.mapflow.Config.minPartitionSize
import io.geoalert.mapflow.Config.sentinelPartitionSize
import io.geoalert.mapflow.Config.zeroArea
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.SourceType

import geotrellis.vector.Extent
import geotrellis.vector.Geometry
import geotrellis.vector.MultiPolygon
import geotrellis.vector.Polygon
import geotrellis.vector.ProjectGeometry
import geotrellis.vector.Projected
import geotrellis.vector.withExtraGeometryMethods

sealed trait Node {
  val bounds: Extent

  def asGeometry: Geometry
}

case class Branch(children: Seq[Leaf]) extends Node {
  override val bounds: Extent = children.map(_.bounds).reduce(_.combine(_))

  def +(leaf: Leaf): Branch =
    Branch(children ++ Seq(leaf))

  override def asGeometry: Geometry = MultiPolygon(children.map(_.value))
}

case class Leaf(value: Polygon) extends Node {
  override val bounds: Extent = value.extent
  override def asGeometry: Geometry = value
}

object BatchingService {
  def partitionGeometry(
      geometry: Projected[Geometry],
      partitionSize: Double,
    ): List[Projected[Geometry]] = {
    def split(
        min: Double,
        max: Double,
        n: Int,
      ): List[(Double, Double)] =
      if (max == min) List[(Double, Double)]()
      else {
        val (minBd, maxBd) = (BigDecimal(min), BigDecimal(max))
        val step = ((maxBd - minBd) / n).setScale(15, BigDecimal.RoundingMode.CEILING)
        val starts = (minBd until maxBd by step).map(_.doubleValue) :+ max
        (starts zip starts.tail).toList
      }

    val poly = geometry.geom
    val extent = geometry.geom.extent
    val nX = (extent.width / partitionSize).ceil.toInt
    val nY = (extent.height / partitionSize).ceil.toInt

    for {
      x <- split(extent.xmin, extent.xmax, nX).par.toList
      y <- split(extent.ymin, extent.ymax, nY).par.toList
      cell = Extent(x._1, y._1, x._2, y._2)
      interGeom <- poly.intersectionSafe(cell).toGeometry().toList
      interPoly = interGeom.buffer(0)
      if interPoly.getArea >= zeroArea // TODO merge with nearby polygons
    } yield interPoly.withSRID(4326)
  }

  def mergeGeometriesToBatches(geometries: Seq[Polygon], partitionSize: Double): Seq[Node] = {
    val leaves = geometries.map(g => Leaf(g))

    if (geometries.isEmpty)
      Seq()
    else
      pack(Seq(), partitionSize, leaves)
  }

  @tailrec
  private def pack(
      acc: Seq[Node],
      partitionSize: Double,
      leaves: Seq[Leaf],
    ): Seq[Node] = {
    val head = leaves.head
    val tail = leaves.tail

    (head, tail) match {
      case (_, Seq()) =>
        acc ++ Seq(head)
      case (head, tail)
           if head.bounds.width > partitionSize || head.bounds.height > partitionSize =>
        pack(acc ++ Seq(head), partitionSize, tail)
      case (head, tail) =>
        // TODO: We can use RTree to optimize this query
        var candidates =
          tail.sortWith((a, b) => head.value.distance(a.value) < head.value.distance(b.value))

        var leaf = Branch(Seq(head))
        while (
            candidates.nonEmpty &&
            leaf.bounds.combine(candidates.head.bounds).width < partitionSize &&
            leaf.bounds.combine(candidates.head.bounds).height < partitionSize
        ) {
          leaf = leaf + candidates.head
          candidates = candidates.tail
        }

        if (candidates.isEmpty)
          acc ++ Seq(leaf)
        else
          pack(acc ++ Seq(leaf), partitionSize, candidates)
    }
  }

  def estimatePartitionSize(processing: Processing): Double = {
    val partitionSizeParam = processing.workflowDef.workflowDefSummary.partitionSize
    val sourceTypeObj = processing.sourceType

    (partitionSizeParam, sourceTypeObj) match {
      case (_, Some(SourceType.sentinel_l2a)) => sentinelPartitionSize
      case (Some(p), _) if p > maxPartitionSize || p < minPartitionSize =>
        throw BadRequest(
          s"Bad partition_size: should be between $minPartitionSize and $maxPartitionSize"
        )
      case (Some(p), _) => p
      case (None, _) => defaultPartitionSize
    }
  }
}
