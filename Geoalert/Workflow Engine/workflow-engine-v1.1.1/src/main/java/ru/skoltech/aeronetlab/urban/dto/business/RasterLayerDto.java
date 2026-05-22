package ru.skoltech.aeronetlab.urban.dto.business;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;

public class RasterLayerDto implements Serializable {

    @JsonIgnore
    private Long workflowId;

    private Long id;

    private String uri;

    public RasterLayerDto() {}

    public RasterLayerDto(Long id, Long workflowId, String uri) {
        this.id = id;
        this.workflowId = workflowId;
        this.uri = uri;
    }

    public Long getId() {
        return this.id;
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

    public Long getWorkflowId() {
        return workflowId;
    }

    public void setWorkflowId(Long workflowId) {
        this.workflowId = workflowId;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("workflowId", workflowId)
                .append("id", id)
                .append("uri", uri)
                .toString();
    }
}
