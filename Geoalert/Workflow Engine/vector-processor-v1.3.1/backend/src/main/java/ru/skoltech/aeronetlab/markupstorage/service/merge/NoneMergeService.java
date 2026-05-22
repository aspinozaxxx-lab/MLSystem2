package ru.skoltech.aeronetlab.markupstorage.service.merge;

import com.vividsolutions.jts.geom.Geometry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureEntity;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureRepository;

import java.util.stream.Stream;

@Service
public class NoneMergeService {

    @Autowired
    private FeatureRepository featureRepository;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    private FeatureEntity cropToMask(FeatureEntity feature, Geometry mask) {
        if (!mask.contains(feature.getGeometry())) {
            feature.setGeometry(feature.getGeometry().intersection(mask));
        }
        return feature;
    }

    private Stream<FeatureEntity> filterAndCrop(Stream<FeatureEntity> features,
                                                Geometry mask,
                                                boolean cropToMask) {
        return features.filter(f -> mask.intersects(f.getGeometry()))
                .map(f -> cropToMask ? cropToMask(f, mask) : f);
    }

    @Transactional
    public void merge(Stream<FeatureEntity> features, ImportParams params, int importId) {
        Stream<FeatureEntity> filteredFeatures = params.getMask()
                .map(m -> filterAndCrop(features, m, params.isCropToMask()))
                .orElse(features)
                .peek(f -> f.getGeometry().setSRID(4326));

        logger.debug("Persisting new features.");

        featureRepository.saveAll(filteredFeatures::iterator);
    }
}
