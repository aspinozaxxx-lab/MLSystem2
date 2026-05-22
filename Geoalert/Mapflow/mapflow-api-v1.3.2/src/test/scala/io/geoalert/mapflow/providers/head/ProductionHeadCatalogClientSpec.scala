package io.geoalert.mapflow.providers.head

import org.scalatest.{FunSpec, Matchers}

class ProductionHeadCatalogClientSpec extends FunSpec with Matchers{

  describe ("HeadImage") {
    it("should generate preview URL") {
      val headImage = new HeadImage("JL1KF01B_PMS01_20220524164030_200085621_104_0001_001_L1",
        Seq((27.21559999999999846, 52.20029999999999859), (27.58500000000000085, 52.14439999999999742),
          (27.49099999999999966, 51.91310000000000002),
          (27.12340000000000018, 51.96880000000000166),
          (27.21559999999999846, 52.20029999999999859)),
        0,
        0,
        0,
        0,
        1,
        "JL1KF01B",
        "2022-05-24T16:40:32.000",
        "unknown")
      headImage.previewUrl should be ("https://home.sat-imagery.com/geoserver/rs_data/wms?SERVICE=WMS&VERSION=1.1.0&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=true&LAYERS=rs_data%3AJL1KF01B_PMS01_20220524164030_200085621_104_0001_001_L1&STYLES=&serverType=geoserver&crossOrigin=anonymous&tiled=false&angle=0&WIDTH=513&HEIGHT=519&SRS=EPSG%3A3857&BBOX=3019363.0765822763%2C6784428.045439347%2C3070748.1535324515%2C6836423.606808357")
    }
  }
}
