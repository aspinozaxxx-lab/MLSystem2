package io.geoalert.vectortileserver

trait Config {

  /** A factor used for filtering features when zoom <= `simplify_max_zoom`. Higher value means less features. */
  val defaultMinAreaFactor: Double = sys.env.getOrElse("DEFAULT_MIN_AREA_FACTOR", "0.0000001").toDouble

  /** A factor used for geometry simplification when zoom <= `simplify_max_zoom` */
  val defaultSimplifyFactor: Double = sys.env.getOrElse("DEFAULT_SIMPLIFY_FACTOR", "0.0006").toDouble

  /** Simplification and filtering are performed when zoom <= `simplify_max_zoom` */
  val defaultSimplifyMaxZoom: Int = sys.env.getOrElse("DEFAULT_SIMPLIFY_MAX_ZOOM", "17").toInt

  /** An empty tile will be returned when zoom < `min_zoom` */
  val defaultMinZoom: Int = sys.env.getOrElse("DEFAULT_MIN_ZOOM", "14").toInt

  /** At most `max_features` features will be rendered per tile */
  val defaultMaxFeatures: Int = sys.env.getOrElse("DEFAULT_MAX_FEATURES", "10000").toInt

  /** URL to this app, accessible from the internet */
  val externalUrl: String = sys.env.getOrElse("EXTERNAL_URL", "http://localhost:8600")

  val dbPort: Int = sys.env.getOrElse("DATABASE_PORT", "5432").toInt
  val dbHost: String = sys.env.get("DATABASE_HOST")
    .orElse(sys.env.get("DATABASE_URI")) // legacy configs
    .getOrElse("vector-database")
  val dbName: String = sys.env.getOrElse("DATABASE_NAME", "vector_db")
  val dbUser: String = sys.env.getOrElse("DATABASE_USER", "postgres")
  val dbPassword:  = sys.env.getOrElse("DATABASE_PASSWORD", "1234Qq")
  val dbSchema: String = sys.env.getOrElse("DATABASE_SCHEMA", "public")
  val dbUrl: String = s"jdbc:postgresql://$dbHost:$dbPort/$dbName?prepareThreshold=0"
}
