package io.geoalert.rastertileserver

object Config {

  /** The port of the HTTP server */
  val port:Int = sys.env.getOrElse("PORT", "8080").toInt

  /** URL of this app, accessible from the web */
  val externalUrl:String = sys.env.getOrElse("EXTERNAL_URL", "http://localhost:8500")

  /** Period in seconds after which attribute store cache will check whether the data is up to date */
  val attributeStoresCacheEvictionPeriod:Int = sys.env.getOrElse("ATTRIBUTE_STORE_CACHE_EVICTION_PERIOD", "120").toInt

  /** Maximum tif sources per tile (when reading tiles from a multi-COG layer) */
  val maxSourcesPerTile:Int = sys.env.getOrElse("MAX_SOURCES_PER_TILE", "8").toInt

  /** Minio settings */
  val minioHost:String = sys.env.getOrElse("MINIO_HOST", "")
  val minioPort:String = sys.env.getOrElse("MINIO_PORT", "")
  val minioEndpoint:String = sys.env.getOrElse("MINIO_ENDPOINT", s"https://$minioHost:$minioPort")
  val minioAccessKey: = sys.env.getOrElse("MINIO_ACCESS_KEY", "")
  val minioSecretKey: = sys.env.getOrElse("MINIO_SECRET_KEY", "")

  /** GDAL settings */
  val vsiCacheSize:String = sys.env.getOrElse("VSI_CACHE_SIZE", "250000000")
  val cplVsilCurlCacheSize:String = sys.env.getOrElse("CPL_VSIL_CURL_CACHE_SIZE", "250000000")

  /**
    * Use JNI GDAL binding. If false Geotrellis pure Java implementation will be used
    */
  val useGdalLibrary: Boolean = sys.env.getOrElse("GDAL", "true").toBoolean

  /**
    * Size of the thread pool for blocking GDAL connection
    */
  val gdalThreadPoolSize: Int = sys.env.getOrElse("GDAL_THREAD_POOL_SIZE", "50").toInt
}
