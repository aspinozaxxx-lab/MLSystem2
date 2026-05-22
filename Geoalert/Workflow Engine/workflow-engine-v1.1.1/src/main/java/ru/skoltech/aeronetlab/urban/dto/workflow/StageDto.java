package ru.skoltech.aeronetlab.urban.dto.workflow;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import org.apache.commons.lang3.builder.ToStringBuilder;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class StageDto implements Serializable {

    private Long id;

    private String name;

    private String description;

    @JsonIgnore
    private Long workflowId;

    private String status;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String errorMessage;

    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<MessageDto> messages = Collections.emptyList();

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private List<Long> taskIds;

    private LocalDateTime statusUpdateDate;

    public StageDto() {}

    public StageDto(Long id, String name, String description, Long workflowId,
                    StageStatus ss) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.workflowId = workflowId;
        this.status = ss.getStatus().toString();
        this.errorMessage = ss.getErrorMessage();
        this.statusUpdateDate = ss.getUpdateDate();
        this.messages = ss.getMessages().stream()
                .map(message -> new MessageDto(message.getCode(), message.getParameters(), message.getMessage()))
                .collect(Collectors.toList());
    }

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

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Long getWorkflowId() {
        return workflowId;
    }

    public void setWorkflowId(Long workflowId) {
        this.workflowId = workflowId;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public List<Long> getTaskIds() {
        return taskIds;
    }

    public void setTaskIds(List<Long> taskIds) {
        this.taskIds = taskIds;
    }

    public LocalDateTime getStatusUpdateDate() {
        return statusUpdateDate;
    }

    public void setStatusUpdateDate(LocalDateTime statusUpdateDate) {
        this.statusUpdateDate = statusUpdateDate;
    }

    public List<MessageDto> getMessages() {
        return messages;
    }

    public void setMessages(List<MessageDto> messages) {
        this.messages = messages;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("id", id)
                .append("name", name)
                .append("description", description)
                .append("workflowId", workflowId)
                .append("status", status)
                .append("errorMessage", errorMessage)
                .append("messages", messages)
                .append("taskIds", taskIds)
                .append("statusUpdateDate", statusUpdateDate)
                .toString();
    }
}
