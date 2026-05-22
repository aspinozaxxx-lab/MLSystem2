package ru.skoltech.aeronetlab.markupstorage.service;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.markupstorage.dto.Feature;
import ru.skoltech.aeronetlab.markupstorage.exception.FeatureCollectionNotFoundException;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureCollectionRepository;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureRepository;

import javax.persistence.EntityManager;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Service
public class FeatureService {

    @Autowired
    private FeatureRepository featureRepository;

    @Autowired
    private FeatureCollectionRepository featureCollectionRepository;

    @Autowired
    private FeatureMapper mapper;

    @Autowired
    private EntityManager entityManager;

    public Stream<Feature> getFeatures(UUID collectionId, Optional<Geometry> area, boolean asPoints) {
        area.ifPresent(g -> g.setSRID(4326));

        if (!featureCollectionRepository.existsById(collectionId)) {
            throw new FeatureCollectionNotFoundException(collectionId);
        }

        Stream<Feature> result = area.map(a -> featureRepository.findInAreaReadOnly(collectionId, a))
                .orElseGet(() -> featureRepository.findAllReadOnly(collectionId))
                .peek(entityManager::detach)
                .map(mapper::fromEntity);

        if (asPoints) {
            result = result.map(this::asPoint);
        }

        return result;
    }

    private Feature asPoint(Feature feature) {
        Geometry poly = feature.getGeometry();
        Point centroid = poly.getCentroid();
        Point point;
        if (centroid.intersects(poly)) {
            point =  centroid;
        } else {
            point = poly.getInteriorPoint();
        }
        feature.setGeometry(point);
        return feature;
    }
}
