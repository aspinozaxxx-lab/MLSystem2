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

import java.util.UUID;
import java.util.stream.Stream;

@Service
public class InstanceSegMergeService extends NoneMergeService {

    @Autowired
    private FeatureRepository featureRepository;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Transactional
    @Override
    public void merge(Stream<FeatureEntity> features, ImportParams params, int importId) {
        super.merge(features, params, importId);

        UUID layerId = params.getLayerId().get();

        params.getMask().ifPresent(m -> deleteOldFeatures(layerId, m, importId));
    }

    private void deleteOldFeatures(UUID layerId, Geometry mask, int importId) {
        logger.debug("Deleting old features within mask.");
        featureRepository.deleteOldWithinArea(layerId, mask, importId);

        logger.debug("Deleting overlapping old features.");
        featureRepository.deleteOldOverlapping(layerId, importId, mask);
    }
}
