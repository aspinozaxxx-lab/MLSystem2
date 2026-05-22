package io.geoalert.rastertileserver.s3

import java.net.URI
import java.time.Instant

import com.typesafe.scalalogging.LazyLogging
import geotrellis.spark.store.hadoop.geotiff.InMemoryGeoTiffAttributeStore
import geotrellis.spark.store.s3.geotiff.S3IMGeoTiffAttributeStore
import geotrellis.store.s3.AmazonS3URI

import scala.collection.concurrent.TrieMap
import scala.jdk.CollectionConverters._
import io.geoalert.rastertileserver.Config._


/** This implementation does not guarantee consistent data at any given moment.
  * It also produces excessive overhead due to checking S3 contents periodically
  * (see the explanation below). A more consistent and optimized solution would be
  * to update the cache in an event-driven fashion. For instance, using
  * Minio/S3 notifications. However, the current solution is eventually consistent,
  * and requires a lot less effort to implement compared to the event-driven one.
  *
  * This implementation caches AttributeStores using the URI as a key.
  * Any new entry that is put into the cache is treated as up-to-date
  * during the next `ATTRIBUTE_STORE_CACHE_EVICTION_PERIOD` seconds.
  * During that period of time the AttributeStore for this URI will always
  * be served directly from the cache. After that period of time the cache entry
  * will become a subject for recheck on the next request. Upon this check,
  * if the set of files in the AttributeStore will be equal to the actual
  * set of files in S3 directory, then the entry will again be treated
  * as up-to-date for the next `ATTRIBUTE_STORE_CACHE_EVICTION_PERIOD` seconds.
  * Otherwise, a new AttributeStore will be created and put into cache.
  * */
object AttributeStoreCache extends LazyLogging {

  private case class CacheEntry(date: Instant, attributeStore: InMemoryGeoTiffAttributeStore)

  private val cache = TrieMap[URI, CacheEntry]()

  def get(uri: URI): InMemoryGeoTiffAttributeStore = {
//    val validUri = uri.trim

    uri.synchronized {
      removeIfOutdated(uri)
      cache.getOrElseUpdate(uri, createCacheEntry(uri)).attributeStore
    }
  }

  /** If the cache entry's date is outdated, checks whether it is still consistent with the list of actual S3 files.
    * If it is indeed consistent, refreshes the entry's date. Otherwise, removes the entry from cache.
    */
  private def removeIfOutdated(uri: URI):Unit = {
    cache.get(uri)
      .filter(_.date.plusSeconds(attributeStoresCacheEvictionPeriod.toLong).isBefore(Instant.now))
      .foreach(expireIfNotUpToDate)

    def expireIfNotUpToDate(entry: CacheEntry): Option[CacheEntry] = {
      val s3Uri = new AmazonS3URI(uri)
      val s3Files = MinioS3Client.listObjects(uri.toString)
        .iterator()
        .asScala
        .map(r => s"s3://${s3Uri.getBucket}/${r.get().objectName()}")
        .filter(s => s.endsWith(".tif") || s.endsWith(".tiff"))
        .toSet
      val cacheFiles = entry.attributeStore.metadataList.map(_.uri.toString).toSet

      if (s3Files == cacheFiles) {
        logger.debug(s"Cache entry out of date but still consistent. Refreshing the date field. URI: $uri")
        cache.replace(uri, entry.copy(date = Instant.now()))
      } else {
        logger.debug(s"Removing inconsistent cache entry. URI: $uri")
        cache.remove(uri)
      }
    }
  }

  private def createCacheEntry(uri: URI): CacheEntry = {
    logger.info(s"Creating a new attribute store for URI: $uri")
    // pattern to get only tiff files, otherwise attributestore fails
    val store = S3IMGeoTiffAttributeStore(uri.toString, uri, ".+\\.(tif|TIF|tiff|TIFF)$", recursive=true, MinioS3Client.s3Client)
    CacheEntry(Instant.now, store)
  }
}
