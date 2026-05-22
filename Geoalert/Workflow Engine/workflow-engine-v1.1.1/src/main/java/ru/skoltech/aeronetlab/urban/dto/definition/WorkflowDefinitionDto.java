package ru.skoltech.aeronetlab.urban.dto.definition;

import java.util.ArrayList;
import java.util.List;

public class WorkflowDefinitionDto {

    private Long id;

    private String name;

    private WorkflowDefinitionVerDto latestVersion;

    private List<WorkflowDefinitionVerDto> versions = new ArrayList<>();

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

    public WorkflowDefinitionVerDto getLatestVersion() {
        return latestVersion;
    }

    public void setLatestVersion(WorkflowDefinitionVerDto latestVersion) {
        this.latestVersion = latestVersion;
    }

    public List<WorkflowDefinitionVerDto> getVersions() {
        return versions;
    }
}
