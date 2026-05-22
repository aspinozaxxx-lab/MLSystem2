package ru.skoltech.aeronetlab.urban.entity.definition;


import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;

import java.util.List;

@Entity
public class WorkflowDefinition {

    @Id
    @GeneratedValue
    private Long id;

    @Column
    private String name;

    @OneToMany(mappedBy = "workflowDefinition")
    private List<WorkflowDefinitionVer> versions;

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

    public List<WorkflowDefinitionVer> getVersions() {
        return versions;
    }

    public void setVersions(List<WorkflowDefinitionVer> versions) {
        this.versions = versions;
    }

    @Override
    public String toString() {
        return String.format("WorkflowDefinition(id=%s)", this.id);
    }
}
