import java.nio.file.Paths
name := "vector-tile-server"

version := sys.env.getOrElse("VERSION", "development")

scalaVersion := "2.13.8"

scalacOptions --= Seq("-Ywarn-value-discard")

resolvers ++= Seq(
  "locationtech-releases" at "https://repo.locationtech.org/content/groups/releases",
  "locationtech-snapshots" at "https://repo.locationtech.org/content/groups/snapshots",
  "Azavea Public Builds" at "https://dl.bintray.com/azavea/geotrellis"
)

libraryDependencies ++= Seq(
  "com.typesafe.akka" %% "akka-http"   % "10.2.10",
  "com.typesafe.akka" %% "akka-http-spray-json" % "10.2.10",
  "com.typesafe.akka" %% "akka-stream" % "2.6.20",
  "com.typesafe.akka" %% "akka-actor" % "2.6.20",
  "de.heikoseeberger" %% "akka-http-circe" % "1.35.3",
  "ch.megard" %% "akka-http-cors" % "1.1.3",
  "org.locationtech.geotrellis" %% "geotrellis-s3-spark" % "3.6.2",
  "org.tpolecat" %% "doobie-core"     % "0.9.4",
  "org.tpolecat" %% "doobie-postgres" % "0.9.4",
  "org.tpolecat" %% "doobie-hikari" % "0.9.4",
  "org.postgresql" % "postgresql" % "42.5.0",
  "org.locationtech.spatial4j" % "spatial4j" % "0.8",
  "ch.qos.logback" % "logback-classic" % "1.4.3",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5"
)

enablePlugins(JavaAppPackaging)
enablePlugins(AshScriptPlugin)

dockerBaseImage := "openjdk:17.0.2-jdk-slim"
dockerExposedPorts := Seq(8080)

val repositoryName = sys.env
  .getOrElse("CI_REGISTRY_IMAGE", "lab.bftcom.com/nspd/geoalert/mapflow")

dockerRepository := Some(Paths.get(repositoryName).getParent.toString)
dockerUpdateLatest := true
Docker / packageName := "vector-tile-server"