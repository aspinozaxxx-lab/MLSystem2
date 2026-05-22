package ru.skoltech.aeronetlab.urban.entity.business;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;

@Entity
public class RasterLayer {

    @Id
    @GeneratedValue
    private Long id;

    /**
     * S3 URI
     */
    private String uri;

    public RasterLayer() {}

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getUri() {
        return uri;
    }

    public void setUri(String uri) {
        this.uri = uri;
    }

    @Override
    public String toString() {
        return String.format("RasterLayer(id=%s)",
                this.id);
    }
}
