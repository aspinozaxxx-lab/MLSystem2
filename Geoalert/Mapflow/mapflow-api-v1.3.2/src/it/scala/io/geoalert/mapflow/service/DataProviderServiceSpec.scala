package io.geoalert.mapflow.service

import cats.syntax.option._
import doobie.implicits._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.model.CreateDataProviderInput
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.UpdateDataProviderInput
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.UserUtil

class DataProviderServiceSpec extends DbIntegrationTest with Services {
  def createDataProvider(name: String): DataProvider = dataProviderService
    .createDataProvider(
      CreateDataProviderInput(
        name,
        name,
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        None,
        0,
        None,
        None,
        None,
        isDefault = false,
        None,
      )
    )(UserUtil.admin)
    .transact(xa)
    .unsafeRunSync()

  describe("DataProviderService") {
    it("should create data provider") {
      val dp = dataProviderService
        .createDataProvider(
          CreateDataProviderInput(
            "mapbox",
            "Mapbox V2",
            "https://mapbox.org/{z}/{x}/{y}.png",
            "https://mapbox.org/{z}/{x}/{y}.png".some,
            0.001,
            "my_mapbox_account".some,
            "my_mapbox_password".some,
            "my_mapbox_token".some,
            isDefault = true,
            None,
          )
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      dp.name should be("mapbox")
      dp.urlTemplate should be(Some("https://mapbox.org/{z}/{x}/{y}.png"))
      dp.previewUrl should be(Some("https://mapbox.org/{z}/{x}/{y}.png"))
      dp.pricePerMp should be(0.001)
      dp.credentialsPassword should be(Some("my_mapbox_password"))
      dp.credentialsUsername should be(Some("my_mapbox_account"))
      dp.credentialsToken should be(Some("my_mapbox_token"))
      dp.isDefault should be(true)
    }

    it("admin should retrieve Data Provider") {
      val existing = createDataProvider("OSM")
      val dp = dataProviderService
        .getDataProvider(existing.id)(UserUtil.admin)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()

      dp.name should be("OSM")
    }

    it("user should see linked Data Provider") {
      val existing = createDataProvider("OSM")
      dataProviderService
        .linkDataProvider(UserUtil.regularUser.id, existing.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val dp = dataProviderService
        .getDataProvider(existing.id)(UserUtil.regularUser)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()

      dp.name should be("OSM")
    }

    it("regular user should not see Data Provider") {
      val existing = createDataProvider("OSM")

      val dp = dataProviderService
        .getDataProvider(existing.id)(UserUtil.regularUser)
        .transact(xa)
        .value
        .unsafeRunSync()

      dp should be(Left(AccessDenied(s"Access denied to Data Provider ${existing.id}")))
    }

    it("should list Data Providers") {
      createDataProvider("OSM")
      val dps = dataProviderService
        .listDataProviders()(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      dps.map(_.name) should contain("OSM")
    }

    it("should update Data Providers") {
      val existing = createDataProvider("OSM")
      val dp = dataProviderService
        .updateDataProvider(
          UpdateDataProviderInput(
            existing.id,
            "new_name".some,
            "New name".some,
            "htto://example.com/{z}/{x}/{y}x2.webp".some,
            None,
            11.0.some,
            "user".some,
            "pass".some,
            "token".some,
            true.some,
            None,
          )
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      dp.name should be("new_name")
    }
  }

  describe("linking data providers") {
    it("should link Data Provider to a user") {
      val existing1 = createDataProvider("OSM 1")
      val existing2 = createDataProvider("OSM 2")
      val user = UserUtil.regularUser
      dataProviderService
        .linkDataProvider(user.id, existing1.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      dataProviderService
        .linkDataProvider(user.id, existing2.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      // Operation should be idempotent
      dataProviderService
        .linkDataProvider(user.id, existing2.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val updatedUser = userService
        .getUsers(Seq(user.id).some, None, None)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()
        .head

      updatedUser.availableDataProviders should contain(existing1)
      updatedUser.availableDataProviders should contain(existing2)
      updatedUser.availableDataProviders.filterNot(_.isDefault).size should be(2)
    }

    it("should unlink Data Provider from a user") {
      val existing1 = createDataProvider("OSM 1")
      val existing2 = createDataProvider("OSM 2")
      val user = UserUtil.regularUser
      dataProviderService
        .linkDataProvider(user.id, existing1.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      dataProviderService
        .linkDataProvider(user.id, existing2.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      // Operation should be idempotent
      dataProviderService
        .unlinkDataProvider(user.id, existing1.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val updatedUser = userService
        .getUsers(Seq(user.id).some, None, None)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()
        .head

      updatedUser.availableDataProviders.filterNot(_.isDefault).size should be(1)
    }
  }

  describe("Well-known data providers") {
    it("should detect mapbox data provider") {
      val url =
        "https://api.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?token=}"
      val existing1 = createDataProvider("Mapbox")
      val user = UserUtil.regularUser

      dataProviderService
        .linkDataProvider(user.id, existing1.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()
      val provider = dataProviderService
        .extractDataProviderFromUrl(url)
        .transact(xa)
        .unsafeRunSync()

      provider should matchPattern {
        case Some(_) =>
      }

      provider.get.name should be("Mapbox")
    }

  }
  it("should detect head data provider") {
    val url =
      "https://app.mapflow.ai/tiles/satimagery/{z}/{x}/{y}.png?year=2022"
    val existing1 = createDataProvider("HEAD")
    val user = UserUtil.regularUser

    dataProviderService
      .linkDataProvider(user.id, existing1.id)(UserUtil.admin)
      .transact(xa)
      .unsafeRunSync()
    val provider = dataProviderService
      .extractDataProviderFromUrl(url)
      .transact(xa)
      .unsafeRunSync()

    provider should matchPattern {
      case Some(_) =>
    }

    provider.get.name should be("HEAD")
  }
}
