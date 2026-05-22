import java.nio.file.Paths
name := "raster-tile-server"

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
  "ch.megard" %% "akka-http-cors" % "1.1.3",
  "com.github.blemale" %% "scaffeine" % "5.2.1",
  "org.locationtech.geotrellis" %% "geotrellis-raster" % "3.6.3",
  "org.locationtech.geotrellis" %% "geotrellis-s3-spark" % "3.6.3",
  "org.locationtech.geotrellis" %% "geotrellis-gdal" % "3.6.3",
  "org.gdal" % "gdal" % "3.5.0",
  "org.apache.spark" %% "spark-core" % "3.3.1",
  "io.minio" % "minio" % "8.4.5",
  "io.prometheus" % "simpleclient_common" % "0.16.0",
  "commons-io" % "commons-io" % "2.11.0",
  "ch.qos.logback" % "logback-classic" % "1.4.4",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5"
)

enablePlugins(JavaAppPackaging)
enablePlugins(AshScriptPlugin)

dockerBaseImage := "aeronetlab/openjdk-gdal:3.1-jdk11-slim"
dockerExposedPorts := Seq(8080)

val repositoryName = sys.env
  .getOrElse("CI_REGISTRY_IMAGE", "lab.bftcom.com/nspd/mapflow")
dockerRepository := Some(Paths.get(repositoryName).getParent.toString)
dockerUpdateLatest := true
Docker / packageName  := "raster-tile-server"
