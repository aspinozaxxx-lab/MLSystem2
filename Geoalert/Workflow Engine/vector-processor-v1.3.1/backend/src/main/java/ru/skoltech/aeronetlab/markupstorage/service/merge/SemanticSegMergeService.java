package ru.skoltech.aeronetlab.markupstorage.service.merge;

import com.vividsolutions.jts.geom.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureEntity;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureRepository;
import ru.skoltech.aeronetlab.markupstorage.service.GeometryUtil;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Service
public class SemanticSegMergeService {

    @Autowired
    private FeatureRepository featureRepository;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    private GeometryFactory factory = new GeometryFactory();

    private FeatureEntity cropToMask(FeatureEntity feature, Geometry mask) {
        if (!mask.contains(feature.getGeometry())) {
            feature.setGeometry(feature.getGeometry().intersection(mask));
        }
        return feature;
    }

    @Transactional
    public void merge(Stream<FeatureEntity> features, ImportParams params, int importId) {
        Geometry mask = params.getMask().orElseThrow(() ->
                new RuntimeException("Mask needs to be specified for SEMANTIC_SEGMENTATION"));
        UUID layerId = params.getLayerId().get();

        logger.debug("Deleting old features within mask.");

        featureRepository.deleteOldWithinArea(layerId, mask, importId);

        Geometry bufferedMask = mask.buffer(0.000001);
        features = features.filter(f -> mask.intersects(f.getGeometry()))
                    .peek(f -> f.setGeometry(GeometryUtil.tryBuffer0(f.getGeometry())))
                    .map(f -> params.isCropToMask() ? cropToMask(f, bufferedMask) : f);

        List<FeatureEntity> old = featureRepository.findInArea(layerId, mask).collect(Collectors.toList());
        List<FeatureEntity> toDelete = new ArrayList<>();
        List<FeatureEntity> merged = new ArrayList<>();

        logger.debug("Merging with " + old.size() + " old features.");

        features.forEach(feature -> {
            List<Integer> mergedIndexes = mergeWithList(feature, merged, params.getKeyProperties());
            for (int i = 0; i < mergedIndexes.size(); i ++) {
                merged.remove(mergedIndexes.get(i) - i);
            }

            List<Integer> oldIndexes = mergeWithList(feature, old, params.getKeyProperties());
            for (int i = 0; i < oldIndexes.size(); i ++) {
                toDelete.add(old.get(oldIndexes.get(i) - i));
                old.remove(oldIndexes.get(i) - i);
            }

            Geometry geom = feature.getGeometry();
            if (geom instanceof GeometryCollection && !(geom instanceof MultiPolygon)) {
                logger.debug("Trying to convert GeometryCollection feature to polygons: " + geom.toText());
                List<Polygon> polygons = GeometryUtil.extractPolygons((GeometryCollection) geom);
                if (polygons.size() == 1) {
                    feature.setGeometry(polygons.get(0));
                    merged.add(feature);
                } else if (polygons.size() > 1) {
                    feature.setGeometry(new MultiPolygon(polygons.toArray(new Polygon[0]), factory));
                    merged.add(feature);
                } else {
                    logger.warn("Skipping GeometryCollection feature containing no polygons: " + geom.toText());
                }
            } else {
                merged.add(feature);
            }
        });

        logger.debug("Deleting " + toDelete.size() + " old features.");
        featureRepository.deleteAll(toDelete);

        merged.forEach(f -> f.getGeometry().setSRID(4326));
        logger.debug("Persisting " + merged.size() + " new features.");
        featureRepository.saveAll(merged);
    }

    // Returns indexes of otherFs that have been merged.
    // As a side-effect changes thisF's geometry in-place.
    private List<Integer> mergeWithList(FeatureEntity thisF,
                                        List<FeatureEntity> otherFs,
                                        Set<String> keyProperties) {
        List<Integer> mergedIndexes = new ArrayList<>();
        for (int i = 0; i < otherFs.size(); i++) {
            FeatureEntity otherF = otherFs.get(i);

            Map<String, Object> thisAts = thisF.getAttributes();
            Map<String, Object> otherAts = otherF.getAttributes();

            boolean keyPropertiesMismatch = keyProperties.stream().anyMatch(
                    p -> thisAts.get(p) == null ? otherAts.get(p) != null : !thisAts.get(p).equals(otherAts.get(p))
            );
            if (keyPropertiesMismatch) continue;

            if (!thisF.getGeometry().intersects(otherF.getGeometry())) continue;

            thisF.setGeometry(thisF.getGeometry().union(GeometryUtil.tryBuffer0(otherF.getGeometry())));
            mergedIndexes.add(i);
        }
        return mergedIndexes;
    }
}
