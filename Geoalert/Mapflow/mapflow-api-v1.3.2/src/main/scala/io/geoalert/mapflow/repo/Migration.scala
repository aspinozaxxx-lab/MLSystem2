package io.geoalert.mapflow.repo

import scala.concurrent.duration._
import Db.xa
import cats.instances.list._
import cats.syntax.traverse._
import doobie.implicits._
import doobie.util.fragment.Fragment.const
import io.geoalert.mapflow.DefaultDbConfig
import io.geoalert.mapflow.TestEnvConfig
import io.geoalert.mapflow.repo.ProcessingRepo.dbSchema
import org.flywaydb.core.Flyway

object Migration extends TestEnvConfig with DefaultDbConfig {
  def flyway(testMode: Boolean): Flyway = {
    val baseLocations = Array("db/migration")

    val testDataLocations = testData.map(d => s"db/test-data/$d").toArray

    Flyway
      .configure
      .dataSource(dbUrlMigration, dbUser, dbPassword)
      .schemas(dbSchema)
      .locations(baseLocations ++ testDataLocations: _*)
      .baselineVersion(if (testMode) "000.0" else "001.0")
      .load
  }

  def apply(): Unit =
    flyway(testMode = false).migrate()
  def resetDb(): Unit = {
    def dropTable(table: String) = const(s"DROP TABLE $dbSchema.$table CASCADE").update.run

    def tables =
      fr"SELECT tablename FROM pg_tables WHERE schemaname = $dbSchema AND tablename <> 'spatial_ref_sys'"
        .query[String]
        .to[List]

    val io = for {
      tables <- tables
      _ <- tables.traverse(dropTable)
    } yield ()
    io.transact(xa).unsafeRunTimed(10.seconds)

    flyway(true).baseline()
    flyway(true).migrate()
  }
}
