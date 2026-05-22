package io.geoalert.mapflow.providers.maxar

import org.apache.commons.codec.Charsets
import org.scalatest.{FunSpec, Matchers}

class DiscoveryApiMetadataParserSpec extends FunSpec with Matchers {
  describe("maxar Discovery API response parser") {
    it("should parse actual Maxar response") {
      val json = new String(getClass.getResourceAsStream("discovery_metadata.json").readAllBytes(), Charsets.UTF_8)
      val meta = DiscoveryApiResponseParser.parseResponse(json)

      meta should matchPattern {
        case Some(DiscoveryApiMetadata(50.539753000000005,160.3864,314.8741,22.238484999999997, "WV03")) =>
      }
    }
  }

}
