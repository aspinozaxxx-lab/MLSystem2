package io.geoalert.mapflow.providers.maxar

import cats.syntax.option._
import geotrellis.vector.JTS.Polygon

import java.time.Instant

object ManualMaxarCatalogClientTest {
  def main(args: Array[String]): Unit = {
    val client = new ProductionMaxarCatalogClient()

    val request = MaxarCatalogRequest(Polygon(Seq(
      (-73.95817300214094,40.801590197725204),
      (-73.98314992701506,40.76789867179309),
      (-73.97209571376592,40.76323560522215),
      (-73.94774845926688,40.79719431737831),
      (-73.95817300214094,40.801590197725204))).some,
      Instant.parse("2020-01-01T00:00:00Z").some,
      None,
      0.1.some,
      None,
      0.01.some,
      None
    )

    val (maxarUsername, maxarPassword, connectId) = (args(0), args(1), args(2))

    val io = client.searchMeta(maxarUsername, maxarPassword, connectId, request)
    val images = io.unsafeRunSync()

    for {
      image <- images
    } yield println(image)

    val imageId = images.head.id

    val singleFeatureRequest = MaxarCatalogRequest(featureId=imageId.some)

    val imageMeta = client.searchMeta(maxarUsername, maxarPassword, connectId, singleFeatureRequest)
      .unsafeRunSync()

    val legacyId = imageMeta.head.metadata.legacyId
    val metadata = client.getDiscoveryApiMetadata(legacyId).unsafeRunSync()
    println(metadata)

  }

}
