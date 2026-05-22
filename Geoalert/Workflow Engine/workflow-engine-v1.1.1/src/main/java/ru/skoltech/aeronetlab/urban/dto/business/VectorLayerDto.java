package ru.skoltech.aeronetlab.urban.dto.business;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;
import java.util.UUID;

public class VectorLayerDto implements Serializable {

    private Long id;

    private UUID layerId;

    @JsonIgnore
    private Long workflowId;

    public VectorLayerDto() {}

    public VectorLayerDto(Long id, UUID layerId, Long workflowId) {
        this.id = id;
        this.workflowId = workflowId;
        this.layerId = layerId;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getWorkflowId() {
        return workflowId;
    }

    public void setWorkflowId(Long workflowId) {
        this.workflowId = workflowId;
    }

    public UUID getLayerId() {
        return layerId;
    }

    public void setLayerId(UUID layerId) {
        this.layerId = layerId;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("id", id)
                .append("layerId", layerId)
                .append("workflowId", workflowId)
                .toString();
    }
}
