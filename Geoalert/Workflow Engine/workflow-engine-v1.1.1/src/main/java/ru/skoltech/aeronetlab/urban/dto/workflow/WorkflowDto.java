package ru.skoltech.aeronetlab.urban.dto.workflow;

import com.fasterxml.jackson.annotation.JsonInclude;
import org.apache.commons.lang3.builder.ToStringBuilder;
import ru.skoltech.aeronetlab.urban.dto.business.AreaOfInterestDto;
import ru.skoltech.aeronetlab.urban.dto.business.RasterLayerDto;
import ru.skoltech.aeronetlab.urban.dto.business.ArtifactDto;
import ru.skoltech.aeronetlab.urban.dto.business.VectorLayerDto;
import ru.skoltech.aeronetlab.urban.entity.workflow.BlockParameters;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WorkflowDto implements Serializable {
    private Long id;

    private Long workflowDefinitionId;

    private List<StageDto> stages = new ArrayList<>();

    private List<BlockParameters> blocks = new ArrayList<>();

    private List<AreaOfInterestDto> areasOfInterest = new ArrayList<>();

    private RasterLayerDto rasterLayer;

    private VectorLayerDto vectorLayer;

    private List<ArtifactDto> artifacts = new ArrayList<>();

    private String status;

    private LocalDateTime statusUpdateDate;

    private LocalDateTime createDate;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String system;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String processingId;

    private Map<String, String> params = new HashMap<>();

    private Map<String, String> meta = new HashMap<>();

    public WorkflowDto(){}

    public WorkflowDto(Long id, LocalDateTime createDate,
                       LocalDateTime statusUpdateDate, StatusType status,
                       Long workflowDefinitionId,
                       String system, String processingId,
                       Map<String, String> params, Map<String, String> meta) {
        this.id = id;
        this.createDate = createDate;
        this.statusUpdateDate = statusUpdateDate;
        this.status = status.toString();
        this.workflowDefinitionId = workflowDefinitionId;
        this.system = system;
        this.processingId = processingId;
        this.params = params;
        this.meta = meta;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public List<AreaOfInterestDto> getAreasOfInterest() {
        return areasOfInterest;
    }

    public List<StageDto> getStages() {
        return stages;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public LocalDateTime getStatusUpdateDate() {
        return statusUpdateDate;
    }

    public void setStatusUpdateDate(LocalDateTime statusUpdateDate) {
        this.statusUpdateDate = statusUpdateDate;
    }

    public Long getWorkflowDefinitionId() {
        return workflowDefinitionId;
    }

    public void setWorkflowDefinitionId(Long workflowDefinitionId) {
        this.workflowDefinitionId = workflowDefinitionId;
    }

    public RasterLayerDto getRasterLayer() {
        return rasterLayer;
    }

    public void setRasterLayer(RasterLayerDto rasterLayer) {
        this.rasterLayer = rasterLayer;
    }

    public VectorLayerDto getVectorLayer() {
        return vectorLayer;
    }

    public void setVectorLayer(VectorLayerDto vectorLayer) {
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

    public List<ArtifactDto> getArtifacts() {
        return artifacts;
    }

    public List<BlockParameters> getBlocks() {
        return blocks;
    }

    public void setBlocks(List<BlockParameters> blocks) {
        this.blocks = blocks;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("id", id)
                .append("workflowDefinitionId", workflowDefinitionId)
                .append("stages", stages)
                .append("areasOfInterest", areasOfInterest)
                .append("rasterLayer", rasterLayer)
                .append("vectorLayer", vectorLayer)
                .append("artifacts", artifacts)
                .append("status", status)
                .append("statusUpdateDate", statusUpdateDate)
                .append("createDate", createDate)
                .append("system", system)
                .append("processingId", processingId)
                .append("params", params)
                .append("meta", meta)
                .toString();
    }
}
