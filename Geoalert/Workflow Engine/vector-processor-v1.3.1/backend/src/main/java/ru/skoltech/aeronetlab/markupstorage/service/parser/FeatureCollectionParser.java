package ru.skoltech.aeronetlab.markupstorage.service.parser;

import ru.skoltech.aeronetlab.markupstorage.dao.FeatureCollectionEntity;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.UUID;

public class FeatureCollectionParser implements AutoCloseable {

    private GeojsonParser geojsonParser;

    public FeatureCollectionParser(InputStream is) throws IOException, ParseGeojsonException {

        geojsonParser = new GeojsonParser(is);
    }

    public FeatureCollectionEntity getFeatureCollection() throws IOException {

        Map<String, String> attributes = geojsonParser.getStringAttributes();

        FeatureCollectionEntity collection = new FeatureCollectionEntity();
        collection.setId(UUID.randomUUID());
        collection.setName(attributes.get("name"));

        return collection;
    }

    @Override
    public void close() throws IOException {

        if (geojsonParser != null) {
            geojsonParser.close();
        }
    }
}
