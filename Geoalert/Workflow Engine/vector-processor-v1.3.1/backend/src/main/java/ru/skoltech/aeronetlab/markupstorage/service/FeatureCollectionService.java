package ru.skoltech.aeronetlab.markupstorage.service;

import com.vividsolutions.jts.geom.Geometry;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureCollectionEntity;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.exception.FeatureCollectionNotFoundException;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureCollectionRepository;
import ru.skoltech.aeronetlab.markupstorage.service.importing.ImportFeatureCollectionService;
import ru.skoltech.aeronetlab.markupstorage.service.importing.ImportFeatureService;

import javax.transaction.Transactional;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
public class FeatureCollectionService {

    @Autowired
    private FeatureCollectionRepository repository;

    @Autowired
    private ImportFeatureCollectionService importCollectionService;

    @Autowired
    private ImportFeatureService importFeatureService;

    @Autowired
    private DownloadGeojsonService downloadGeojsonService;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public void deleteFeatureCollection(UUID collectionId) {

        if (!repository.existsById(collectionId)) {
            throw new FeatureCollectionNotFoundException(collectionId);
        }

        repository.deleteById(collectionId);
    }

    public FeatureCollectionEntity getFeatureCollection(UUID collectionId) {

        return repository.findById(collectionId).orElseThrow(() -> new FeatureCollectionNotFoundException(collectionId));
    }

    public List<FeatureCollectionEntity> getAllFeatureCollections() {

        List<FeatureCollectionEntity> collections = new ArrayList<>();
        repository.findAll().forEach(collections::add);

        return collections;
    }

    public FeatureCollectionEntity postFromUrl(String url, ImportParams params)
            throws IOException, ParseGeojsonException {

        List<Path> geojsons = downloadGeojsonService.download(url, true);

        if (geojsons.size() == 0) {
            throw new IllegalArgumentException("No geojson files supplied."); //TODO exception handling, should be 400
        }

        FeatureCollectionEntity collection = importCollectionService.importFromPath(geojsons.get(0), params);
        params.setLayerId(Optional.of(collection.getId()));

        importFeatureService.importFromPaths(geojsons, params, registerNextImportId(collection));

        return collection;
    }

    public FeatureCollectionEntity postFromMultipartFile(MultipartFile file, ImportParams params)
            throws IOException, ParseGeojsonException {

        if (!"geojson".equalsIgnoreCase(FilenameUtils.getExtension(file.getOriginalFilename()))) {
            throw new IllegalArgumentException("Unsupported file type."); //TODO exception handling, should be 400
        }

        FeatureCollectionEntity collection = importCollectionService.importFromMultipartFile(file, params);
        params.setLayerId(Optional.of(collection.getId()));

        importFeatureService.importFromMultipartFile(file, params, registerNextImportId(collection));

        return collection;
    }

    public FeatureCollectionEntity postFromS3(String bucket, String fileName, ImportParams params)
            throws IOException, ParseGeojsonException {

        FeatureCollectionEntity collection = importCollectionService.importFromS3(bucket, fileName, params);
        params.setLayerId(Optional.of(collection.getId()));

        importFeatureService.importFromS3(bucket, fileName, params, registerNextImportId(collection));

        return collection;
    }

    @Transactional
    public void updateExtent(UUID collectionId, int importId) {
        repository.updateExtent(collectionId, importId);
    }

    @Transactional
    protected int registerNextImportId(FeatureCollectionEntity collection) {
        logger.debug("Registering next importId for feature collection " + collection.getId());

        int importId = collection.getLastImportId() + 1;

        collection.setLastImportId(importId);
        repository.save(collection);

        logger.debug("Registered importId =" + importId + " for feature collection " + collection.getId());

        return importId;
    }
}
