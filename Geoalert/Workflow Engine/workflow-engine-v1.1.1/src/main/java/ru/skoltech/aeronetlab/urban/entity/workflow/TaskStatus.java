package ru.skoltech.aeronetlab.urban.entity.workflow;

import com.vladmihalcea.hibernate.type.basic.PostgreSQLHStoreType;
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
import org.hibernate.annotations.Type;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Entity
@Table(indexes = @Index(name = "task_status__task_id_idx", columnList = "task_id"))
public class TaskStatus {

    @Id
    @GeneratedValue
    private Long id;

    @OneToOne
    private Task task;

    @Enumerated(EnumType.ORDINAL)
    private StatusType status;

    @Column(columnDefinition = "TEXT")
    private String message;

    @ElementCollection(fetch = FetchType.EAGER)
    private List<Message> messages = new ArrayList<>();

    private LocalDateTime updateDate;

    private LocalDateTime startDate;

    private Integer attempts;

    @Type(PostgreSQLHStoreType.class)
    @Column(columnDefinition = "hstore")
    private Map<String, String> results = Collections.emptyMap();

    private static final List<StatusType> LEGAL_STATUSES =
            Arrays.asList(StatusType.FAILED, StatusType.IN_PROGRESS, StatusType.OK, StatusType.WAITING_TO_RETRY);

    public TaskStatus() {}

    public TaskStatus(Task task) {
        this.task = task;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Task getTask() {
        return task;
    }

    public void setTask(Task task) {
        this.task = task;
    }

    public StatusType getStatus() {
        return status;
    }

    public void setStatus(StatusType status) {
        assert(LEGAL_STATUSES.contains(status));
        this.status = status;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
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

    public Integer getAttempts() {
        return attempts;
    }

    public void setAttempts(Integer attempts) {
        this.attempts = attempts;
    }

    public List<Message> getMessages() {
        return messages;
    }

    public void setMessages(List<Message> messages) {
        this.messages = messages;
    }

    public Map<String, String> getResults() {
        return results;
    }

    public void setResults(Map<String, String> results) {
        this.results = results;
    }

    @Override
    public String toString() {
        return String.format("TaskStatus(id=%s; task=%s; status=%s)",
                this.id,
                this.task,
                this.status);
    }
}
