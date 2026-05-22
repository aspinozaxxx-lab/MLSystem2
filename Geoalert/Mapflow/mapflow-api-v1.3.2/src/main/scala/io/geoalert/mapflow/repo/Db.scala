package io.geoalert.mapflow.repo

import java.util.concurrent.Executors
import scala.concurrent.ExecutionContext
import cats.effect.Blocker
import cats.effect.ContextShift
import cats.effect.IO
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.zaxxer.hikari.HikariConfig
import com.zaxxer.hikari.HikariDataSource
import doobie.hikari.HikariTransactor
import io.geoalert.mapflow.DefaultDbConfig

object Db {
  lazy val xa: HikariTransactor[IO] = Transactor.xa
}

object Transactor extends DefaultDbConfig {
  /*
    Execution context for awaiting connections from DB. If application starts to fail on obtaining DB connection,
    this pool SHOULD be increased
   */
  private lazy val connectionEC =
    ExecutionContext.fromExecutor(
      ExecutionContext.fromExecutor(
        Executors.newFixedThreadPool(
          connectionPoolSize,
          new ThreadFactoryBuilder().setNameFormat("db-connection-%d").build(),
        )
      )
    )

  /*
    Execution context for performing DB transactions
   */
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
        Executors.newFixedThreadPool(
          connectionPoolSize,
          new ThreadFactoryBuilder().setNameFormat("db-client-%d").build(),
        )
      )
    )

  private lazy val hikariDataSource = {
    val hikariConfig = new HikariConfig()
    hikariConfig.setPoolName("mapflow-hikari-pool")
    hikariConfig.setMaximumPoolSize(connectionPoolSize)
    hikariConfig.setJdbcUrl(dbUrl)
    hikariConfig.setUsername(dbUser)
    hikariConfig.setPassword(dbPassword)
    hikariConfig.setDriverClassName("org.postgresql.Driver")
    hikariConfig.setLeakDetectionThreshold(leakDetectionThresholdMs)

    new HikariDataSource(hikariConfig)
  }

  lazy val blocker: Blocker = Blocker.liftExecutionContext(transactionEC)

  lazy val xa: HikariTransactor[IO] = {
    implicit val contextShift: ContextShift[IO] = cs
    HikariTransactor.apply[IO](
      hikariDataSource,
      connectionEC,
      blocker,
    )
  }
}
