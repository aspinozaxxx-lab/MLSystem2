package io.geoalert.mapflow

import io.geoalert.mapflow.repo.Migration
import io.geoalert.mapflow.rest.json.Decoders
import io.geoalert.mapflow.rest.json.Encoders
import org.scalatest.BeforeAndAfter
import org.scalatest.FunSpec
import org.scalatest.Matchers

trait DbIntegrationTest
    extends FunSpec
       with BeforeAndAfter
       with Matchers
       with Encoders
       with Decoders {
  before {
    Migration.resetDb()
  }
}
