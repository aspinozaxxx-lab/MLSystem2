package ru.skoltech.aeronetlab.urban.entity.workflow;

import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.ManyToMany;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;

import java.util.HashSet;
import java.util.Set;

@Entity
@Table(indexes = @Index(name = "stage__workflow_id_idx", columnList = "workflow_id"))
public class Stage {

    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne
    private Workflow workflow;

    @ManyToOne
    private StageDefinition stageDefinition;

    @ManyToMany(fetch = FetchType.EAGER)
    private Set<Stage> previousStages = new HashSet<>();

    @OneToOne(mappedBy = "stage")
    private StageStatus stageStatus;

    public Stage() {}

    public Stage(Set<Stage> previousStages) {
        this.previousStages.addAll(previousStages);
    }

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

    public StageDefinition getStageDefinition() {
        return stageDefinition;
    }

    public void setStageDefinition(StageDefinition stageDefinition) {
        this.stageDefinition = stageDefinition;
    }

    public Set<Stage> getPreviousStages() {
        return previousStages;
    }

    public StageStatus getStageStatus() {
        return stageStatus;
    }

    public void setStageStatus(StageStatus stageStatus) {
        this.stageStatus = stageStatus;
    }

    @Override
    public String toString() {
        return String.format("Stage(id=%s; name=%s; workflow=%s)",
                this.id,
                this.stageDefinition.getName(),
                this.workflow);
    }
}
