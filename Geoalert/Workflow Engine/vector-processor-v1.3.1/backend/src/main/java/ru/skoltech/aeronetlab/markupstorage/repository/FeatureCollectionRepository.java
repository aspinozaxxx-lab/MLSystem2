package ru.skoltech.aeronetlab.markupstorage.repository;

import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureCollectionEntity;

import java.util.UUID;

public interface FeatureCollectionRepository extends CrudRepository<FeatureCollectionEntity, UUID> {

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(nativeQuery = true, value = "update {h-schema}layer " +
            "set extent = st_envelope(st_union(" +
            "  (select st_setsrid(coalesce(st_envelope(st_extent(geometry)), st_geomfromtext('POLYGON EMPTY')), 4326) " +
            "  from {h-schema}feature where layer_id = ?1 and import_id = ?2), " +
            "  (select st_setsrid(coalesce(extent, st_geomfromtext('POLYGON EMPTY')), 4326) from {h-schema}layer where id = ?1) " +
            ")) where id = ?1")
    void updateExtent(UUID collectionId, int importId);
}
