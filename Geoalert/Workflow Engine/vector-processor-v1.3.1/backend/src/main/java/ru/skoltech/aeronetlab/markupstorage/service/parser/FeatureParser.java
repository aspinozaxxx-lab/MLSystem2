package ru.skoltech.aeronetlab.markupstorage.service.parser;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.skoltech.aeronetlab.markupstorage.dto.Feature;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;

import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

public class FeatureParser implements AutoCloseable, Iterator<Feature> {

    private static final int MAX_ATTEMPTS = 1_000;

    private GeojsonParser geojsonParser;

    private Feature current = null;

    private long count = 0;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public FeatureParser(InputStream is) throws IOException, ParseGeojsonException {
        geojsonParser = new GeojsonParser(is);

        if (geojsonParser.goToStartOfArray("features")) {
            current = getFeature();
        }
    }

    private Feature getFeature() {
        int attempt = 0;

        while (true) {
            try {
                if (attempt > MAX_ATTEMPTS) {
                    logger.error("Exceeded max consecutive attempts to read a feature.");
                    return null;
                }

                Feature feature = geojsonParser.nextObject(Feature.class);

                if (feature != null) {
                    feature.getGeometry().setSRID(4326);
                    count++;
                }

                return feature;
            } catch (IOException e) {
                attempt++;
                logger.error("Couldn't parse feature #" + count + ", error message is: " + e.getMessage());
            }
        }
    }

    public boolean hasNext() {
        return current != null;
    }

    public Feature next() {
        Feature feature = current;
        if (current != null) {
            current = getFeature();
        }
        return feature;
    }

    public long getCount() {
        return count;
    }

    @Override
    public void close() throws IOException {
        if (geojsonParser != null) {
            geojsonParser.close();
        }
    }
}
