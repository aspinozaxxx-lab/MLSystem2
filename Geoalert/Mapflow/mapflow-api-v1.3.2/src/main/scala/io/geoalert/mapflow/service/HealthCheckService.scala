package io.geoalert.mapflow.service

import doobie.free.connection.ConnectionIO
import io.geoalert.mapflow.repo.UserRepo

class HealthCheckService {
  def heartbeat(): ConnectionIO[String] = for {
    _ <- UserRepo.healthCheck()
  } yield "OK"
}

object HealthCheckService {
  def apply(): HealthCheckService = new HealthCheckService()
}
