package ru.skoltech.aeronetlab.urban.entity.workflow;

import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;

@Entity
@Table(indexes = @Index(name = "workflow_status__workflow_id_idx", columnList = "workflow_id"))
public class WorkflowStatus {

    @Id
    @GeneratedValue
    private Long id;

    @OneToOne
    private Workflow workflow;

    @Enumerated(EnumType.ORDINAL)
    private StatusType status;

    private LocalDateTime updateDate;

    private static final List<StatusType> LEGAL_STATUSES =
            Arrays.asList(StatusType.CANCELLED, StatusType.FAILED, StatusType.IN_PROGRESS, StatusType.OK);

    public WorkflowStatus() {}

    public WorkflowStatus(Workflow workflow) {
        this.workflow = workflow;
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

    public StatusType getStatus() {
        return status;
    }

    public void setStatus(StatusType status) {
        assert(LEGAL_STATUSES.contains(status));
        this.status = status;
    }

    public LocalDateTime getUpdateDate() {
        return updateDate;
    }

    public void setUpdateDate(LocalDateTime updateDate) {
        this.updateDate = updateDate;
    }

    @Override
    public String toString() {
        return String.format("WorkflowStatus(id=%s; workflow=%s; status=%s)",
                this.id,
                this.workflow,
                this.status);
    }
}
