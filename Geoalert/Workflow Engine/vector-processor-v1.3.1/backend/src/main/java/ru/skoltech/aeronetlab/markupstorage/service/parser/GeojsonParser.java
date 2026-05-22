package ru.skoltech.aeronetlab.markupstorage.service.parser;

import com.bedatadriven.jackson.datatype.jts.JtsModule;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

public class GeojsonParser implements AutoCloseable {

    private JsonParser jsonParser;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public GeojsonParser(InputStream is) throws IOException, ParseGeojsonException {

        JsonFactory jsonFactory = new JsonFactory();
        jsonParser = jsonFactory.createParser(is);

        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JtsModule());
        jsonParser.setCodec(mapper);

        if (jsonParser.nextToken() != JsonToken.START_OBJECT) {
            throw new ParseGeojsonException("Geojson must start with '{'.");
        }
    }

    public boolean goToStartOfArray(String fieldName) throws IOException {

        while (jsonParser.nextToken() != null) {
            if (fieldName.equals(jsonParser.getCurrentName())) {
                if (jsonParser.nextToken() == JsonToken.START_ARRAY) {
                    return true;
                }
            }
        }

        return false;
    }

    public Map<String, String> getStringAttributes() throws IOException {

        Map<String, String> map = new HashMap<>();

        if (jsonParser.getCurrentToken() != JsonToken.START_OBJECT) {
            return map;
        }

        while (jsonParser.nextToken() != null) {
            if (jsonParser.getCurrentToken() == JsonToken.VALUE_STRING) {
                map.put(jsonParser.getCurrentName(), jsonParser.getValueAsString());
            } else if (jsonParser.getCurrentToken() == JsonToken.END_OBJECT) {
                return map;
            } else {
                jsonParser.skipChildren();
            }
        }

        return map;
    }

    public <T> T nextObject(Class<T> type) throws IOException {
        try {
            if (jsonParser.nextValue() == JsonToken.START_OBJECT) {
                T obj = jsonParser.readValueAs(type);
                return obj;
            } else {
                return null;
            }
        } catch (IOException e) {
            try {
                jsonParser.readValueAsTree();
            } catch (Exception any) {}
            throw e;
        }
    }

    @Override
    public void close() throws IOException {

        if (jsonParser != null) {
            jsonParser.close();
        }
    }
}
