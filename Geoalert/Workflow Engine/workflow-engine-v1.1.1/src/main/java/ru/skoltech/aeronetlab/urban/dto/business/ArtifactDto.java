package ru.skoltech.aeronetlab.urban.dto.business;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import org.apache.commons.lang3.builder.ToStringBuilder;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;

import java.io.Serializable;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class ArtifactDto implements Serializable {

    private Long areaOfInterestId;

    private String artifactType;

    private String uri;

    @JsonIgnore
    private Long workflowId;

    public ArtifactDto() {}

    public ArtifactDto(Long areaOfInterestId, ArtifactType artifactType, String uri, Long workflowId) {
        this.areaOfInterestId = areaOfInterestId;
        this.artifactType = artifactType.toString();
        this.uri = uri;
        this.workflowId = workflowId;
    }

    public Long getAreaOfInterestId() {
        return areaOfInterestId;
    }

    public void setAreaOfInterestId(Long areaOfInterestId) {
        this.areaOfInterestId = areaOfInterestId;
    }

    public String getArtifactType() {
        return artifactType;
    }

    public void setArtifactType(String artifactType) {
        this.artifactType = artifactType;
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
                .append("areaOfInterestId", areaOfInterestId)
                .append("artifactType", artifactType)
                .append("uri", uri)
                .append("workflowId", workflowId)
                .toString();
    }
}
