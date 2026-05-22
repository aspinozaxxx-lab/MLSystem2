package ru.skoltech.aeronetlab.markupstorage.service.export;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vividsolutions.jts.geom.Geometry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.markupstorage.dto.Feature;
import ru.skoltech.aeronetlab.markupstorage.service.FeatureService;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.Optional;
import java.util.UUID;

@Service
public class StreamingExportGeojsonService {

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private FeatureService featureService;

    @Transactional(readOnly = true)
    public void pipeGeojson(UUID collectionId, Optional<Geometry> mask, OutputStream os,
                            boolean asPoints, boolean wrapInFc)
            throws IOException {
        if (wrapInFc) {
            os.write("{\"type\":\"FeatureCollection\",\"features\":[".getBytes());
        }

        Iterator<Feature> iter = featureService.getFeatures(collectionId, mask, asPoints).iterator();

        if (iter.hasNext()) {
            writeFeature(os, iter.next());
        }

        while (iter.hasNext()) {
            os.write(", ".getBytes());
            writeFeature(os, iter.next());
        }

        if (wrapInFc) {
            os.write("]}".getBytes());
        }
    }

    private void writeFeature(OutputStream os, Feature feature) {
        try {
            os.write(objectMapper.writeValueAsBytes(feature));
        } catch (IOException e) {
            throw new RuntimeException("Error exporting geojson : " + e, e);
        }
    }
}
