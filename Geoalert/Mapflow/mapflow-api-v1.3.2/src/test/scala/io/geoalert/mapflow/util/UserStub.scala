package io.geoalert.mapflow.util

import cats.syntax.option._
import io.geoalert.mapflow.model.{BillingType, DataProvider, Role, User}

import java.time.Instant
import java.util.UUID

trait UserStub {
  val admin: User = User(UUID.fromString("61cd6899-19e8-44a0-97db-b86f1a9b7af4"),
    "admin@example.com", Role.Admin, 100, 100, BillingType.Area,
    Instant.parse("2020-01-01T00:10:10Z"), Instant.parse("2020-01-01T00:10:10Z"), 0, 100000, 10, List(
      DataProvider(UUID.randomUUID(), "securewatch", "SecureWatch Standard", None, None,
        10, "maxar_user".some, "maxar_pass".some, "SOME_CONNECT_ID".some, isDefault = false, None)
    ), none, reviewWorkflowEnabled = false, None, None, None)

  val regularUser: User = User(UUID.fromString("71cd6899-19e8-44a0-97db-b86f1a9b7af5"),
    "bb855a4b-e109-410c-b00d-8455ba6af790", Role.User, 100, 100, BillingType.Area,
    Instant.parse("2020-01-01T00:10:10Z"), Instant.parse("2020-01-01T00:10:10Z"), 0, 100000, 10, List(
      DataProvider(UUID.randomUUID(), "securewatch", "SecureWatch Standard", None, None,
        10, "maxar_user".some, "maxar_pass".some, "SOME_CONNECT_ID".some, isDefault = false, None)
    ), none, reviewWorkflowEnabled = false, None, None, None)

  val premiumUser: User = User(UUID.fromString("9acd6899-19e8-44a0-97db-b86f1a9b7a77"),
    "premium@example.com", Role.User, 100, 100, BillingType.Area,
    Instant.parse("2020-01-01T00:10:10Z"), Instant.parse("2020-01-01T00:10:10Z"), 0, 100000, 10, List(
      DataProvider(UUID.randomUUID(), "securewatch", "SecureWatch Standard", None, None,
        10, "maxar_user".some, "maxar_pass".some, "SOME_CONNECT_ID".some, isDefault = false, None)
    ), none, reviewWorkflowEnabled = false, None, None, None)
}
