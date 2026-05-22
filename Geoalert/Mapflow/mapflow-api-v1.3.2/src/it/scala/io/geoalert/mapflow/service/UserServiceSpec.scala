package io.geoalert.mapflow.service

import cats.syntax.option._
import doobie.implicits._

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.CreateUserInput
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.UpdateUserInput
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.UserUtil

class UserServiceSpec extends DbIntegrationTest with Services {
  describe("User Service") {
    it("should create user") {
      val userIo = userService.createUser(
        CreateUserInput(
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          Role.User.some,
          "123qwe321".some,
          42L.some,
          56L.some,
          BillingType.Area,
          None,
          None,
          None,
        )
      )(UserUtil.admin)
      val user = userIo.transact(xa).unsafeRunSync()

      user.memoryLimit should be(1_000_000_000)
      user.email should be("97228df6-5dd9-482e-8e9a-8bd98067b21e")
      user.processedArea should be(0)
      user.areaLimit should be(42)
      user.aoiAreaLimit should be(56)
      user.role should be(Role.User)
      user.billingType should be(BillingType.Area)
      user.maxAoisPerProcessing should be(10)
    }

    it("should update user") {
      UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e", Role.User.some, "123qwe321".some)
      val userIo = userService.updateUser(
        UpdateUserInput(
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          Role.Admin.some,
          none,
          1000L.some,
          500L.some,
          BillingType.Credits.some,
          500_000_000L.some,
          25.some,
          None,
          None,
        )
      )(UserUtil.admin)
      val user = userIo.transact(xa).unsafeRunSync()

      user.memoryLimit should be(500_000_000)
      user.maxAoisPerProcessing should be(25)
      user.billingType should be(BillingType.Credits)
    }
  }
}
