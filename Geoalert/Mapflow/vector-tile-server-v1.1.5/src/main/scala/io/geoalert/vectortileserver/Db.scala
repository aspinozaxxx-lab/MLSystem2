package io.geoalert.vectortileserver

import java.util.concurrent.Executors

import cats.effect.{Blocker, ContextShift, IO}
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.zaxxer.hikari.{HikariConfig, HikariDataSource}
import doobie.hikari.HikariTransactor

import scala.concurrent.ExecutionContext

trait Db extends Config {
  lazy val xa: HikariTransactor[IO] = Transactor.xa
}

object Transactor extends Config {
  private lazy val connectionEC =
    ExecutionContext.fromExecutor(
      ExecutionContext.fromExecutor(
        Executors.newFixedThreadPool(8, new ThreadFactoryBuilder().setNameFormat("db-connection-%d").build())
      )
    )

  private lazy val transactionEC =
    ExecutionContext.fromExecutor(
      ExecutionContext.fromExecutor(
        Executors.newCachedThreadPool(
          new ThreadFactoryBuilder()
            .setNameFormat("db-transaction-%d")
            .build()
        )
      )
    )

  lazy val cs: ContextShift[IO] =
    IO.contextShift(
      ExecutionContext.fromExecutor(
        Executors.newFixedThreadPool(8, new ThreadFactoryBuilder().setNameFormat("db-client-%d").build())
      )
    )

  private lazy val hikariDataSource = {
    val hikariConfig = new HikariConfig()
    hikariConfig.setPoolName("vector-tile-server-hikari-pool")
    hikariConfig.setMaximumPoolSize(8)
    hikariConfig.setJdbcUrl(dbUrl)
    hikariConfig.setUsername(dbUser)
    hikariConfig.setPassword(dbPassword)
    hikariConfig.setDriverClassName("org.postgresql.Driver")

    new HikariDataSource(hikariConfig)
  }

  lazy val blocker: Blocker = Blocker.liftExecutionContext(transactionEC)

  lazy val xa: HikariTransactor[IO] = {
    implicit val contextShift: ContextShift[IO] = cs
    HikariTransactor.apply[IO](
      hikariDataSource,
      connectionEC,
      blocker
    )
  }
}
