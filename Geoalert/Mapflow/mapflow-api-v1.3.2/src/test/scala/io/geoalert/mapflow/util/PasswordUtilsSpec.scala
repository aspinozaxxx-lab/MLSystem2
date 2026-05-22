package io.geoalert.mapflow.util

import org.scalatest._

class PasswordUtilsSpec extends FunSpec with Matchers {
  describe("PasswordUtils") {
    it("should validate correct password") {
      val hash = PasswordUtils.generatePasswordHash("test")
      val result = PasswordUtils.validatePassword("test", hash)
      result should be (true)
    }

    it("should spot incorrect password") {
      val hash = PasswordUtils.generatePasswordHash("test")
      val result = PasswordUtils.validatePassword("test2", hash)
      result should be (false)
    }
  }
}
