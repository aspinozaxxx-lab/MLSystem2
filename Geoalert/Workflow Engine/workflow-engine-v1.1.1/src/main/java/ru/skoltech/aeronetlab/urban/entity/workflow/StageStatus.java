package ru.skoltech.aeronetlab.urban.entity.workflow;

import jakarta.persistence.Column;
import jakarta.persistence.ElementCollection;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Entity
@Table(indexes = @Index(name = "stage_status__stage_id_idx", columnList = "stage_id"))
public class StageStatus {
    @Id
    @GeneratedValue
    private Long id;

    @OneToOne
    private Stage stage;

    @Enumerated(EnumType.ORDINAL)
    private StatusType status;

    @Column(length = 1024)
    private String errorMessage;

    @ElementCollection(fetch = FetchType.EAGER)
    private List<Message> messages = new ArrayList<>();

    private LocalDateTime updateDate;
    private LocalDateTime startDate;

    private static final List<StatusType> LEGAL_STATUSES = Arrays.asList(
            StatusType.FAILED, StatusType.IN_PROGRESS, StatusType.OK, StatusType.PENDING, StatusType.CANCELLED
    );

    public StageStatus() {}

    public StageStatus(Stage stage) {
        this.stage = stage;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Stage getStage() {
        return stage;
    }

    public void setStage(Stage stage) {
        this.stage = stage;
    }

    public StatusType getStatus() {
        return status;
    }

    public void setStatus(StatusType status) {
        assert(LEGAL_STATUSES.contains(status));
        if (status != StatusType.FAILED) this.errorMessage = null;
        this.status = status;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage == null ? null :
                errorMessage.substring(0, Math.min(1024, errorMessage.length()));
    }

    public LocalDateTime getUpdateDate() {
        return updateDate;
    }

    public void setUpdateDate(LocalDateTime updateDate) {
        this.updateDate = updateDate;
    }

    public LocalDateTime getStartDate() {
        return startDate;
    }

    public void setStartDate(LocalDateTime startDate) {
        this.startDate = startDate;
    }

    public List<Message> getMessages() {
        return messages;
    }

    public void setMessages(List<Message> messages) {
        this.messages = messages;
    }

    @Override
    public String toString() {
        return String.format("StageStatus(id=%s; stage=%s; status=%s)",
                this.id,
                this.stage,
                this.status);
    }
}
