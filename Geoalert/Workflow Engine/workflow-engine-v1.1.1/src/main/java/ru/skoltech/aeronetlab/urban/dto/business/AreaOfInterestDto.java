package ru.skoltech.aeronetlab.urban.dto.business;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.locationtech.jts.geom.Geometry;

import java.io.Serializable;

public class AreaOfInterestDto implements Serializable {

    private Long id;

    private Geometry geometry;

    @JsonIgnore
    private Long workflowId;

    public AreaOfInterestDto() {}

    public AreaOfInterestDto(Long id, Geometry geometry, Long workflowId) {
        this.id = id;
        this.geometry = geometry;
        this.workflowId = workflowId;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Geometry getGeometry() {
        return geometry;
    }

    public void setGeometry(Geometry geometry) {
        this.geometry = geometry;
    }

    public Long getWorkflowId() {
        return workflowId;
    }

    public void setWorkflowId(Long workflowId) {
        this.workflowId = workflowId;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("id", id)
                .append("geometry", geometry)
                .append("workflowId", workflowId)
                .toString();
    }
}
