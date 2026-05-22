package ru.skoltech.aeronetlab.urban.entity.definition;

import com.vladmihalcea.hibernate.type.basic.PostgreSQLHStoreType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToMany;
import jakarta.persistence.ManyToOne;
import org.hibernate.annotations.Type;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Entity
public class StageDefinition {

    @Id
    @GeneratedValue
    private Long id;

    private String name;

    @ManyToOne
    private WorkflowDefinitionVer workflowDefinitionVer;

    @Enumerated(EnumType.STRING)
    private Action action;

    @ManyToMany(fetch = FetchType.EAGER)
    private Set<StageDefinition> previousStages = new HashSet<>();

    private String description;

    private Integer retries;

    private Integer retryInterval;

    @Type(PostgreSQLHStoreType.class)
    @Column(columnDefinition = "hstore")
    private Map<String, String> params = new HashMap<>();

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public WorkflowDefinitionVer getWorkflowDefinitionVer() {
        return workflowDefinitionVer;
    }

    public void setWorkflowDefinitionVer(WorkflowDefinitionVer workflowDefinitionVer) {
        this.workflowDefinitionVer = workflowDefinitionVer;
    }

    public Set<StageDefinition> getPreviousStages() {
        return previousStages;
    }

    public Action getAction() {
        return action;
    }

    public void setAction(Action action) {
        this.action = action;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Map<String, String> getParams() {
        return params;
    }

    public Integer getRetries() {
        return retries;
    }

    public void setRetries(Integer retries) {
        this.retries = retries;
    }

    public Integer getRetryInterval() {
        return retryInterval;
    }

    public void setRetryInterval(Integer retryInterval) {
        this.retryInterval = retryInterval;
    }

    public List<String> getParamAsList(String param) {
        String inputsParam = getParams().getOrDefault(param, "");
        return Arrays.stream(inputsParam.split(","))
                .map(String::trim)
                .distinct()
                .collect(Collectors.toList());
    }

    public void setPreviousStages(Set<StageDefinition> previousStages) {
        this.previousStages = previousStages;
    }

    public void setParams(Map<String, String> params) {
        this.params = params;
    }

    @Override
    public String toString() {
        return String.format("StageDefinition(id=%s; name=%s; workflowDefinitionVer=%s)",
                this.id,
                this.name,
                this.workflowDefinitionVer);
    }
}
