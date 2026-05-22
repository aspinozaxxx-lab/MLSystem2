import java.nio.file.Paths

name := "mapflow-api"

version := sys.env.getOrElse("VERSION", "development")

scalaVersion := "2.13.8"

val akkaHttpVersion = "10.2.9"
val akkaVersion = "2.6.19"
val enumeratumVersion = "1.7.0"

tpolecatScalacOptions ~= { opts =>
  opts.filterNot(Set(ScalacOptions.fatalWarnings, ScalacOptions.warnValueDiscard)) ++ Set(
    ScalacOptions.privatePartialUnification
  )
}

Universal / javaOptions ++= Seq("-J-Xmx4G", "-J-Xms4G")
scalacOptions ++= Seq(
  "-Ymacro-annotations"
)
resolvers ++= Seq(
  "locationtech-releases" at "https://repo.locationtech.org/content/groups/releases",
  "locationtech-snapshots" at "https://repo.locationtech.org/content/groups/snapshots",
  "osgeo-releases" at "https://repo.osgeo.org/repository/release/",
)

libraryDependencies ++= Seq(
  "org.scala-lang.modules"      %% "scala-parallel-collections" % "1.0.4",
  "org.typelevel"               %% "cats-core"                  % "2.8.0",
  "com.typesafe.akka"           %% "akka-http"                  % akkaHttpVersion,
  "com.typesafe.akka"           %% "akka-stream"                % akkaVersion,
  "com.typesafe.akka"           %% "akka-actor"                 % akkaVersion,
  "com.typesafe.akka"           %% "akka-http-xml"              % akkaHttpVersion,
  "com.lightbend.akka"          %% "akka-stream-alpakka-s3"     % "3.0.3",
  "de.heikoseeberger"           %% "akka-http-circe"            % "1.35.3",
  "io.circe"                    %% "circe-core"                 % "0.13.0",
  "io.circe"                    %% "circe-generic"              % "0.13.0",
  "io.circe"                    %% "circe-optics"               % "0.13.0",
  "io.circe"                    %% "circe-yaml"                 % "0.14.1",
  "io.circe"                    %% "circe-generic-extras"       % "0.13.0",
  "com.pauldijou"               %% "jwt-circe"                  % "4.2.0",
  "com.fasterxml.jackson.core"   % "jackson-databind"           % "2.9.8",
  "org.yaml"                     % "snakeyaml"                  % "1.25",
  "ch.megard"                   %% "akka-http-cors"             % "1.1.3",
  "org.sangria-graphql"         %% "sangria"                    % "3.5.3",
  "org.sangria-graphql"         %% "sangria-circe"              % "1.3.2",
  "org.tpolecat"                %% "doobie-core"                % "0.9.4",
  "org.tpolecat"                %% "doobie-postgres"            % "0.9.4",
  "org.tpolecat"                %% "doobie-postgres-circe"      % "0.9.4",
  "org.tpolecat"                %% "doobie-hikari"              % "0.9.4",
  "org.tpolecat"                %% "doobie-quill"               % "0.9.4",
  "io.getquill"                 %% "quill-jdbc"                 % "3.6.0",
  "net.postgis"                  % "postgis-jdbc"               % "2.3.0",
  "io.minio"                     % "minio"                      % "6.0.6",
  "com.amazonaws"                % "aws-java-sdk-s3"            % "1.11.624",
  "org.locationtech.geotrellis" %% "geotrellis-spark"           % "3.6.2",
  "org.locationtech.geotrellis" %% "geotrellis-vector"          % "3.6.2",
  "ch.qos.logback"               % "logback-classic"            % "1.2.3",
  "org.slf4j"                    % "log4j-over-slf4j"           % "1.7.36",
  "org.slf4j"                    % "slf4j-api"                  % "1.7.36",
  "com.typesafe.scala-logging"  %% "scala-logging"              % "3.9.2",
  "org.flywaydb"                 % "flyway-core"                % "6.0.8",
  "com.google.guava"             % "guava"                      % "30.1-jre",
  "com.github.blemale"          %% "scaffeine"                  % "5.2.1",
  "org.testcontainers"           % "postgresql"                 % "1.17.6",
  "com.beachape"                %% "enumeratum"                 % enumeratumVersion,
  "com.beachape"                %% "enumeratum-circe"           % enumeratumVersion,
  "com.beachape"                %% "enumeratum-cats"            % enumeratumVersion,
  "org.scalatest"               %% "scalatest"                  % "3.0.8"         % Test,
  "com.typesafe.akka"           %% "akka-http-testkit"          % akkaHttpVersion % Test,
  "com.typesafe.akka"           %% "akka-stream-testkit"        % akkaVersion     % Test,
)

excludeDependencies ++= Seq(
  ExclusionRule("log4j", "log4j")
)

/** Integration tests */
lazy val IntTest = config("it") extend Test

lazy val mapflowApi =
  Project(id = "mapflow-api", base = file("."))
    .configs(IntTest)
    .settings(Defaults.itSettings: _*)

IntTest / parallelExecution := false
IntTest / fork              := true
IntTest / concurrentRestrictions += Tags.limit(Tags.Test, 1)

enablePlugins(JavaAppPackaging)

dockerBaseImage    := "openjdk:17.0.2-jdk-slim"
dockerExposedPorts := Seq(8080)

addCompilerPlugin("org.typelevel" %% "kind-projector"     % "0.13.2" cross CrossVersion.full)
addCompilerPlugin("com.olegpy"    %% "better-monadic-for" % "0.3.1")

val repositoryName = sys
  .env
  .getOrElse("CI_REGISTRY_IMAGE", "registry.gitlab.com/geoalert-projects")

dockerRepository     := Some(Paths.get(repositoryName).getParent.toString)
dockerUpdateLatest   := true
Docker / packageName := "mapflow-api"
