package ru.skoltech.aeronetlab.markupstorage.repository;

import com.vividsolutions.jts.geom.Geometry;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.jpa.repository.QueryHints;
import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureEntity;

import javax.persistence.QueryHint;
import java.util.UUID;
import java.util.stream.Stream;

import static org.hibernate.annotations.QueryHints.READ_ONLY;
import static org.hibernate.jpa.QueryHints.HINT_CACHEABLE;
import static org.hibernate.jpa.QueryHints.HINT_FETCH_SIZE;

public interface FeatureRepository extends CrudRepository<FeatureEntity, Long> {

    @Query("select f from " +
            "FeatureEntity f " +
            "join f.featureCollection fc " +
            "where fc.id = ?1 and st_intersects(f.geometry, ?2) = true")
    @QueryHints(value = {
            @QueryHint(name = HINT_FETCH_SIZE, value = "100"),
            @QueryHint(name = HINT_CACHEABLE, value = "false"),
            @QueryHint(name = READ_ONLY, value = "true")
    })
    public Stream<FeatureEntity> findInAreaReadOnly(UUID collectionId, Geometry area);

    @Query("select f from " +
            "FeatureEntity f " +
            "join f.featureCollection fc " +
            "where fc.id = ?1 and st_intersects(f.geometry, ?2) = true")
    @QueryHints(value = @QueryHint(name = HINT_FETCH_SIZE, value = "100"))
    public Stream<FeatureEntity> findInArea(UUID collectionId, Geometry area);

    @Query("select f from " +
            "FeatureEntity f " +
            "join f.featureCollection fc " +
            "where fc.id = ?1")
    @QueryHints(value = {
            @QueryHint(name = HINT_FETCH_SIZE, value = "100"),
            @QueryHint(name = HINT_CACHEABLE, value = "false"),
            @QueryHint(name = READ_ONLY, value = "true")
    })
    public Stream<FeatureEntity> findAllReadOnly(UUID collectionId);

    @Query("select f from " +
            "FeatureEntity f " +
            "join f.featureCollection fc " +
            "where fc.id = ?1")
    @QueryHints(value = @QueryHint(name = HINT_FETCH_SIZE, value = "100"))
    public Stream<FeatureEntity> findAll(UUID collectionId);

    @Query(
            value = "delete from {h-schema}feature " +
                    "where layer_id = ?1 and import_id <> ?3 and st_within(geometry, ?2) = true",
            nativeQuery = true
    )
    @Modifying
    public void deleteOldWithinArea(UUID collectionId, Geometry area, int importId);

    @Query(value = "delete from {h-schema}feature where id in " +
            "(select case when st_area(f1.geometry) > st_area(f2.geometry) then f2.id else f1.id end as id " +
            "from {h-schema}feature as f1 " +
            "join {h-schema}feature as f2 " +
            "on st_intersects(f1.geometry, f2.geometry) " +
            "and st_area(st_intersection(f1.geometry, f2.geometry)) > 0.7 * least(st_area(f1.geometry), st_area(f2.geometry)) " +
            "where st_intersects(f1.geometry, ?3)  " +
            "and f1.import_id <> ?2 and f2.import_id = ?2 " +
            "and f1.layer_id = ?1 and f2.layer_id = ?1)", nativeQuery = true)
    @Modifying
    public void deleteOldOverlapping(UUID collectionId, int importId, Geometry area);
}
