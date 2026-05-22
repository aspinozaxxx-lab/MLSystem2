package ru.skoltech.aeronetlab.markupstorage.service.importing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureCollectionEntity;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureEntity;
import ru.skoltech.aeronetlab.markupstorage.dto.Feature;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureCollectionRepository;
import ru.skoltech.aeronetlab.markupstorage.service.FeatureMapper;
import ru.skoltech.aeronetlab.markupstorage.service.merge.InstanceSegMergeService;
import ru.skoltech.aeronetlab.markupstorage.service.merge.MergeStrategy;
import ru.skoltech.aeronetlab.markupstorage.service.merge.NoneMergeService;
import ru.skoltech.aeronetlab.markupstorage.service.merge.SemanticSegMergeService;
import ru.skoltech.aeronetlab.markupstorage.service.parser.FeatureParser;

import java.util.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

@Service
public class TransactionalImportFeatureService {

    @Autowired
    private FeatureCollectionRepository collectionRepository;

    @Autowired
    private FeatureMapper mapper;

    @Autowired
    private InstanceSegMergeService instanceSegMergeService;

    @Autowired
    private SemanticSegMergeService semanticSegMergeService;

    @Autowired
    private NoneMergeService noneMergeService;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    private FeatureEntity toEntity(Feature feature, FeatureCollectionEntity collection, int importId) {
        FeatureEntity featureEntity = mapper.fromDto(feature);
        featureEntity.setFeatureCollection(collection);
        featureEntity.setImportId(importId);
        return featureEntity;
    }

    @Transactional
    public void importFeatures(FeatureParser parser, ImportParams params, int importId) {

        UUID layerId = params.getLayerId().get();

        FeatureCollectionEntity collection = collectionRepository.findById(layerId)
                .orElseThrow(() -> new RuntimeException("Layer with id=" + layerId + " doesn't exist."));

        logger.debug("Creating features stream.");

        Stream<FeatureEntity> features = StreamSupport.stream(((Iterable<Feature>) (() -> parser)).spliterator(), false)
                .filter(f -> f != null && f.getGeometry() != null)
                .map(f -> toEntity(f, collection, importId));

        logger.debug("Merging new features into the feature collection.");

        if (params.getMergeStrategy() == MergeStrategy.INSTANCE_SEGMENTATION) {
            instanceSegMergeService.merge(features, params, importId);
        } else if (params.getMergeStrategy() == MergeStrategy.SEMANTIC_SEGMENTATION) {
            semanticSegMergeService.merge(features, params, importId);
        } else if (params.getMergeStrategy() == MergeStrategy.NONE) {
            noneMergeService.merge(features, params, importId);
        } else {
            throw new RuntimeException("Unsupported merge strategy: " + params.getMergeStrategy());
        }
    }
}
