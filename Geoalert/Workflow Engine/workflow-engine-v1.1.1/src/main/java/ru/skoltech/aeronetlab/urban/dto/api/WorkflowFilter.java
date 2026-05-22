package ru.skoltech.aeronetlab.urban.dto.api;

import java.time.LocalDateTime;
import java.util.Set;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.locationtech.jts.geom.Geometry;

public class WorkflowFilter {

    private Set<Long> workflowIds;
    private Set<Long> workflowDefinitionIds;
    private Set<String> systems;
    private Set<String> processingIds;
    private Set<String> statuses;
    private Geometry geometry;
    private LocalDateTime createDateFrom;
    private LocalDateTime createDateTo;

    public Set<Long> getWorkflowIds() {
        return workflowIds;
    }

    public void setWorkflowIds(Set<Long> workflowIds) {
        this.workflowIds = workflowIds;
    }

    public Set<Long> getWorkflowDefinitionIds() {
        return workflowDefinitionIds;
    }

    public void setWorkflowDefinitionIds(Set<Long> workflowDefinitionIds) {
        this.workflowDefinitionIds = workflowDefinitionIds;
    }

    public Set<String> getSystems() {
        return systems;
    }

    public void setSystems(Set<String> systems) {
        this.systems = systems;
    }

    public Set<String> getProcessingIds() {
        return processingIds;
    }

    public void setProcessingIds(Set<String> processingIds) {
        this.processingIds = processingIds;
    }

    public Set<String> getStatuses() {
        return statuses;
    }

    public void setStatuses(Set<String> statuses) {
        this.statuses = statuses;
    }

    public Geometry getGeometry() {
        return geometry;
    }

    public void setGeometry(Geometry geometry) {
        this.geometry = geometry;
    }

    public LocalDateTime getCreateDateFrom() {
        return createDateFrom;
    }

    public void setCreateDateFrom(LocalDateTime createDateFrom) {
        this.createDateFrom = createDateFrom;
    }

    public LocalDateTime getCreateDateTo() {
        return createDateTo;
    }

    public void setCreateDateTo(LocalDateTime createDateTo) {
        this.createDateTo = createDateTo;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("workflowIds", workflowIds)
                .append("workflowDefinitionIds", workflowDefinitionIds)
                .append("systems", systems)
                .append("processingIds", processingIds)
                .append("statuses", statuses)
                .append("geometry", geometry)
                .append("createDateFrom", createDateFrom)
                .append("createDateTo", createDateTo)
                .toString();
    }
}
