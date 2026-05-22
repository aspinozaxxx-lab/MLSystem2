package ru.skoltech.aeronetlab.markupstorage.service.export;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vividsolutions.jts.geom.Geometry;
import io.minio.MinioClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.markupstorage.dto.FeatureCollection;
import ru.skoltech.aeronetlab.markupstorage.service.FeatureService;

import java.io.*;
import java.util.Optional;
import java.util.UUID;

@Service
public class ExportGeojsonService {

    @Autowired
    private FeatureService featureService;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private MinioClient minioClient;

    @Transactional(readOnly = true)
    public void exportGeojsonToS3(UUID collectionId, Optional<Geometry> mask, String bucket,
                                  String path, boolean asPoints) {
        try {
            byte[] bytes = generateGeojson(collectionId, mask, asPoints);

            if (!minioClient.bucketExists(bucket)) {
                minioClient.makeBucket(bucket);
            }

            try (ByteArrayInputStream is = new ByteArrayInputStream(bytes)) {
                minioClient.putObject(bucket, path, is, MediaType.APPLICATION_JSON.toString());
            }
        } catch (Exception e) {
            throw new RuntimeException("Error exporting geojson to s3: " + e.toString(), e);
        }
    }

    private byte[] generateGeojson(UUID collectionId, Optional<Geometry> mask, boolean asPoints) {

        FeatureCollection featureCollection = new FeatureCollection();

        featureService.getFeatures(collectionId, mask, asPoints).forEach(f -> featureCollection.getFeatures().add(f));

        try {
            return objectMapper.writeValueAsBytes(featureCollection);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Error generating geojson: " + e.toString(), e);
        }
    }
}
