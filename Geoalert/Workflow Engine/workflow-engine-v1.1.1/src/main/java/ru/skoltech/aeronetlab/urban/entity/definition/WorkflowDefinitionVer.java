package ru.skoltech.aeronetlab.urban.entity.definition;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.util.Collection;

@Entity
public class WorkflowDefinitionVer {

    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne
    private WorkflowDefinition workflowDefinition;

    @OneToMany(mappedBy = "workflowDefinitionVer")
    private Collection<StageDefinition> stageDefinitions;

    @JdbcTypeCode(SqlTypes.JSON)
    private Collection<BlockConfig> blockConfig;

    private Integer version;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public WorkflowDefinition getWorkflowDefinition() {
        return workflowDefinition;
    }

    public void setWorkflowDefinition(WorkflowDefinition workflowDefinition) {
        this.workflowDefinition = workflowDefinition;
    }

    public Integer getVersion() {
        return version;
    }

    public void setVersion(Integer version) {
        this.version = version;
    }

    public Collection<StageDefinition> getStageDefinitions() {
        return stageDefinitions;
    }

    public void setStageDefinitions(Collection<StageDefinition> stageDefinitions) {
        this.stageDefinitions = stageDefinitions;
    }

    public Collection<BlockConfig> getBlockConfig() {
        return blockConfig;
    }

    public void setBlockConfigs(Collection<BlockConfig> blockConfig) {
        this.blockConfig = blockConfig;
    }

    @Override
    public String toString() {
        return String.format("WorkflowDefinitionVersion(id=%s; workflowDefinitionVer=%s; version=%s)",
                this.id,
                this.workflowDefinition,
                this.version);
    }
}
