package ru.skoltech.aeronetlab.urban.entity.workflow;

import com.vladmihalcea.hibernate.type.basic.PostgreSQLHStoreType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.annotations.Type;
import org.hibernate.type.SqlTypes;
import ru.skoltech.aeronetlab.urban.entity.business.RasterLayer;
import ru.skoltech.aeronetlab.urban.entity.business.VectorLayer;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;

import java.time.LocalDateTime;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

@Entity
public class Workflow {
    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne
    private WorkflowDefinitionVer workflowDefinitionVer;

    @OneToOne(mappedBy = "workflow")
    private WorkflowStatus workflowStatus;

    @ManyToOne
    private RasterLayer rasterLayer;

    @ManyToOne
    private VectorLayer vectorLayer;

    @Column(length = 64)
    private String system;

    @Column(length = 64)
    private String processingId;

    @Type(PostgreSQLHStoreType.class)
    @Column(columnDefinition = "hstore")
    private Map<String, String> params = new HashMap<>();

    @Type(PostgreSQLHStoreType.class)
    @Column(columnDefinition = "hstore")
    private Map<String, String> meta = new HashMap<>();

    @JdbcTypeCode(SqlTypes.JSON)
    private Collection<BlockParameters> blockParams;

    private LocalDateTime createDate = LocalDateTime.now();

    public Workflow() {}

    public Workflow(WorkflowDefinitionVer workflowDefinitionVer) {
        this.workflowDefinitionVer = workflowDefinitionVer;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public WorkflowDefinitionVer getWorkflowDefinitionVer() {
        return workflowDefinitionVer;
    }

    public void setWorkflowDefinitionVer(WorkflowDefinitionVer workflowDefinitionVer) {
        this.workflowDefinitionVer = workflowDefinitionVer;
    }

    public RasterLayer getRasterLayer() {
        return rasterLayer;
    }

    public void setRasterLayer(RasterLayer rasterLayer) {
        this.rasterLayer = rasterLayer;
    }

    public VectorLayer getVectorLayer() {
        return vectorLayer;
    }

    public void setVectorLayer(VectorLayer vectorLayer) {
        this.vectorLayer = vectorLayer;
    }

    public LocalDateTime getCreateDate() {
        return createDate;
    }

    public void setCreateDate(LocalDateTime createDate) {
        this.createDate = createDate;
    }

    public String getSystem() {
        return system;
    }

    public void setSystem(String system) {
        this.system = system;
    }

    public String getProcessingId() {
        return processingId;
    }

    public void setProcessingId(String processingId) {
        this.processingId = processingId;
    }

    public Map<String, String> getParams() {
        return params;
    }

    public Map<String, String> getMeta() {
        return meta;
    }

    public WorkflowStatus getWorkflowStatus() {
        return workflowStatus;
    }

    public void setWorkflowStatus(WorkflowStatus workflowStatus) {
        this.workflowStatus = workflowStatus;
    }

    public Collection<BlockParameters> getBlockParams() {
        return blockParams;
    }

    public void setBlockParams(Collection<BlockParameters> blockParams) {
        this.blockParams = blockParams;
    }

    @Override
    public String toString() {
        return String.format("Workflow(id=%s)",
                this.id);
    }
}
