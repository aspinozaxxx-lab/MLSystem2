package ru.skoltech.aeronetlab.markupstorage.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.vividsolutions.jts.geom.Geometry;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Feature implements Serializable {

    private final String type = "Feature";

    @JsonProperty("properties")
    private Map<String, Object> properties = new HashMap<>();

    private Geometry geometry;

    public String getType() {
        return type;
    }

    public Geometry getGeometry() {
        return geometry;
    }

    public void setGeometry(Geometry geometry) {
        this.geometry = geometry;
    }

    public Map<String, Object> getProperties() {
        return properties;
    }
}
