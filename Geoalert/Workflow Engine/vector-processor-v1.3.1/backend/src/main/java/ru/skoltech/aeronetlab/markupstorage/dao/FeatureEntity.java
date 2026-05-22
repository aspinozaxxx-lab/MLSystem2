package ru.skoltech.aeronetlab.markupstorage.dao;

import com.vividsolutions.jts.geom.Geometry;
import com.vladmihalcea.hibernate.type.json.JsonBinaryType;
import org.hibernate.annotations.Type;
import org.hibernate.annotations.TypeDef;

import javax.persistence.*;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

@Entity
@Table(name = "feature")
@TypeDef(name = "jsonb", typeClass = JsonBinaryType.class)
public class FeatureEntity implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "feature_sequence")
    @SequenceGenerator(name = "feature_sequence", sequenceName = "hibernate_sequence", allocationSize = 1)
    private Long id;

    private int importId;

    private Long classId;

    @Type(type = "jsonb")
    @Column(columnDefinition = "jsonb")
    private Map<String, Object> attributes = new HashMap<>();

    @Column(name = "geometry", columnDefinition="Geometry")
    private Geometry geometry;

    @ManyToOne
    @JoinColumn(name = "layer_id")
    private FeatureCollectionEntity featureCollection;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getClassId() {
        return classId;
    }

    public void setClassId(Long classId) {
        this.classId = classId;
    }

    public Geometry getGeometry() {
        return geometry;
    }

    public void setGeometry(Geometry geometry) {
        this.geometry = geometry;
    }

    public FeatureCollectionEntity getFeatureCollection() {
        return featureCollection;
    }

    public void setFeatureCollection(FeatureCollectionEntity featureCollection) {
        this.featureCollection = featureCollection;
    }

    public Map<String, Object> getAttributes() {
        return attributes;
    }

    public int getImportId() {
        return importId;
    }

    public void setImportId(int importId) {
        this.importId = importId;
    }
}
