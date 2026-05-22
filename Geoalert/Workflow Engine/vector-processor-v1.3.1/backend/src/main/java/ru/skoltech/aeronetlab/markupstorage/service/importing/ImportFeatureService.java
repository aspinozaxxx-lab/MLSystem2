package ru.skoltech.aeronetlab.markupstorage.service.importing;

import com.vividsolutions.jts.geom.Geometry;
import io.minio.MinioClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;
import ru.skoltech.aeronetlab.markupstorage.service.FeatureCollectionService;
import ru.skoltech.aeronetlab.markupstorage.service.merge.MergeStrategy;
import ru.skoltech.aeronetlab.markupstorage.service.parser.FeatureParser;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
public class ImportFeatureService {

    @Autowired
    private MinioClient minioClient;

    @Autowired
    private FeatureCollectionService featureCollectionService;

    @Autowired
    private TransactionalImportFeatureService transactionalImportFeatureService;

    @Value("${thread.count:4}")
    private int threadCount;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public void importFromPaths(List<Path> paths, ImportParams params, int importId) throws IOException, ParseGeojsonException {

        for (Path path : paths) {
            importFromStream(Files.newInputStream(path), params, importId);
        }
    }

    public void importFromMultipartFile(MultipartFile file, ImportParams params, int importId)
            throws IOException, ParseGeojsonException {

        importFromStream(file.getInputStream(), params, importId);
    }

    public void importFromS3(String bucket, String fileName, ImportParams params, int importId)
            throws IOException, ParseGeojsonException {

        InputStream is = null;
        try {
            logger.debug("Opening input stream to get data from minio, bucket=" + bucket + ", fileName=" + fileName);
            is = minioClient.getObject(bucket, fileName);
            logger.debug("Finished opening input stream to get data from minio");
        } catch (Exception e) {
            throw new RuntimeException("Error getting content from S3: " + e.toString(), e);
        }

        importFromStream(is, params, importId);
    }

    private void importFromStream(InputStream is, ImportParams params, int importId)
            throws IOException, ParseGeojsonException {

        FeatureParser parser = new FeatureParser(is); //TODO close parser after all threads have finished

        logger.info("Starting to import features.");

        transactionalImportFeatureService.importFeatures(parser, params, importId);

        logger.debug("Updating feature collection extent.");

        featureCollectionService.updateExtent(params.getLayerId().get(), importId);
    }
}
