package ru.skoltech.aeronetlab.markupstorage.service.importing;

import io.minio.MinioClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureCollectionEntity;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureCollectionRepository;
import ru.skoltech.aeronetlab.markupstorage.service.parser.FeatureCollectionParser;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.UUID;

@Service
public class ImportFeatureCollectionService {

    @Autowired
    private FeatureCollectionRepository repository;

    @Autowired
    private MinioClient minioClient;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public FeatureCollectionEntity importFromPath(Path path, ImportParams params)
            throws IOException, ParseGeojsonException {

        return importFromStream(Files.newInputStream(path), params);
    }

    public FeatureCollectionEntity importFromMultipartFile(MultipartFile file, ImportParams params)
            throws IOException, ParseGeojsonException {

        return importFromStream(file.getInputStream(), params);
    }

    public FeatureCollectionEntity importFromS3(String bucket, String fileName, ImportParams params)
            throws IOException, ParseGeojsonException {

        InputStream is = null;
        try {
            logger.debug("Opening input stream to get data from minio, bucket=" + bucket + ", fileName=" + fileName);
            is = minioClient.getObject(bucket, fileName);
            logger.debug("Finished opening input stream to get data from minio");
        } catch (Exception e) {
            throw new RuntimeException("Error getting content from S3: " + e.toString(), e);
        }

        return importFromStream(is, params);
    }

    @Transactional
    protected FeatureCollectionEntity importFromStream(InputStream is, ImportParams params) {

        logger.info("Starting to import feature collection.");

        Optional<FeatureCollectionEntity> collectionOptional =
                params.getLayerId().flatMap(i -> repository.findById(i));

        FeatureCollectionEntity fc = collectionOptional.orElseGet(() -> {
            try (FeatureCollectionParser parser = new FeatureCollectionParser(is)) {
                FeatureCollectionEntity collection = parser.getFeatureCollection();

                params.getLayerName().ifPresent(n -> collection.setName(n));
                params.getLayerId().ifPresent(uuid -> collection.setId(uuid));

                collection.setLastImportId(-1);

                return repository.save(collection);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        logger.info("Finished importing feature collection.");

        return fc;
    }
}
