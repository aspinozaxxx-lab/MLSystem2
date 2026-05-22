package ru.skoltech.aeronetlab.markupstorage.dao;

import com.vividsolutions.jts.geom.Geometry;

import javax.persistence.*;
import java.io.Serializable;
import java.util.UUID;

@Entity
@Table(name = "layer")
public class FeatureCollectionEntity implements Serializable {

    @Id
    @Column(columnDefinition = "uuid")
    private UUID id;

    private int lastImportId;

    private String name;

    @Column(name = "extent", columnDefinition="Geometry")
    private Geometry extent;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Geometry getExtent() {
        return extent;
    }

    public int getLastImportId() {
        return lastImportId;
    }

    public void setLastImportId(int lastImportId) {
        this.lastImportId = lastImportId;
    }

    public FeatureCollectionEntity setExtent(Geometry extent) {
        this.extent = extent;
        return this;
    }
}
