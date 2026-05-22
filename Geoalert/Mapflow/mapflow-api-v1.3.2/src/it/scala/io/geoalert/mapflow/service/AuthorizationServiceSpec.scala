package io.geoalert.mapflow.service

import scala.concurrent.Await
import scala.concurrent.duration._
import cats.syntax.option._
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.Migration
import io.geoalert.mapflow.service.AuthorizationService.AvanpostUser
import io.geoalert.mapflow.util.UserUtil
import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSpec
import org.scalatest.Matchers
import pdi.jwt.JwtOptions

class AuthorizationServiceSpec extends FunSpec with Matchers with BeforeAndAfterAll with Services {
  override def beforeAll(): Unit = {
    Migration.resetDb()
    UserUtil.createUser("35206105-350e-41f5-b874-742741358580", Role.User.some, "SoM3Pa33".some)
  }

  describe("Password sign in") {
    it("Should be able to login using login/password pair") {
      val future = authorizationService.authenticate("35206105-350e-41f5-b874-742741358580", "SoM3Pa33")
      val userOpt = Await.result(future, 3.seconds)
      userOpt should matchPattern {
        case Some(_: User) =>
      }
    }
  }

  describe("JWT token") {
    it("should decode JWT token") {
      val user = new AuthorizationService(userService) {
        override val jwtOptions: JwtOptions =
        JwtOptions(signature = false, expiration = false, notBefore = false, 0)
        override val jwtKey: String =
          "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAw/hwmDhppdvtjBuWLvBZKwSEoIOYlZxaQyqAVAYD1yCRzhBRCZ5vByRV2Bd3BbWdn1/44Kq1sn/zvxr3vrN2K4qH5MPTrl2d3ahXAD5d5Is2n+uTSnAa8Pa5F+76hf2qOrVXvimKeBJ1+Fd0RWc6iHEOnwg1ANK0W7ms/3jzDFblr9VK21/jV+Ct0moybYN8ju9VeOu44U0i4H8mkauTf8Fe2ES0Yv38aubSUEt7zschL0Quo5SGIaYEjxDYdaDpwnyf43ZLC8s89Yn0xP58mlPVhwshVaxAF4NF3U8b2ocJhIEJn+pctkbYzjEmRqKRwJJF6Fc2yllB+7OfkFmVKQIDAQAB"
      }.decodeToken(
        "REDACTED_JWT"
      )

      user should matchPattern {
        case Right(AvanpostUser(_, Some("customer name"), Some("customer"))) =>
      }
    }
  }
}
