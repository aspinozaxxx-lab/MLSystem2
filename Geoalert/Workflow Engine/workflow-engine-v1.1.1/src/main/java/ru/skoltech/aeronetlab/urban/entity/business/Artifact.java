package ru.skoltech.aeronetlab.urban.entity.business;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

@Entity
@Table(indexes = {@Index(name = "artifact__workflow_id_idx", columnList = "workflow_id"),
        @Index(name = "artifact__area_of_interest_id_idx", columnList = "area_of_interest_id")})
public class Artifact {

    public Artifact(Workflow workflow, AreaOfInterest areaOfInterest, ArtifactType artifactType, String uri) {
        this.workflow = workflow;
        this.areaOfInterest = areaOfInterest;
        this.artifactType = artifactType;
        this.uri = uri;
    }

    public Artifact(Workflow workflow, ArtifactType artifactType, String uri) {
        this.workflow = workflow;
        this.artifactType = artifactType;
        this.uri = uri;
    }

    public Artifact() {
    }

    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne
    private Workflow workflow;

    @OneToOne
    private AreaOfInterest areaOfInterest;

    @Enumerated(EnumType.STRING)
    private ArtifactType artifactType;

    @Column(length = 1024)
    private String uri;

    @Column(length = 32)
    private String cacheKey;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Workflow getWorkflow() {
        return workflow;
    }

    public void setWorkflow(Workflow workflow) {
        this.workflow = workflow;
    }

    public AreaOfInterest getAreaOfInterest() {
        return areaOfInterest;
    }

    public void setAreaOfInterest(AreaOfInterest areaOfInterest) {
        this.areaOfInterest = areaOfInterest;
    }

    public ArtifactType getArtifactType() {
        return artifactType;
    }

    public void setArtifactType(ArtifactType artifactType) {
        this.artifactType = artifactType;
    }

    public String getUri() {
        return uri;
    }

    public void setUri(String uri) {
        this.uri = uri;
    }

    public String getCacheKey() {
        return cacheKey;
    }

    public void setCacheKey(String cacheKey) {
        this.cacheKey = cacheKey;
    }

    @Override
    public String toString() {
        return String.format("StatisticsData(id=%s; workflow=%s; type=%s)",
                this.id,
                this.workflow,
                this.artifactType);
    }
}
