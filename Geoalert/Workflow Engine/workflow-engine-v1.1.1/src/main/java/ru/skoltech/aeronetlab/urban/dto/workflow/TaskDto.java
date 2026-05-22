package ru.skoltech.aeronetlab.urban.dto.workflow;

import com.fasterxml.jackson.annotation.JsonInclude;
import org.apache.commons.lang3.builder.ToStringBuilder;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;

import java.time.LocalDateTime;

public class TaskDto {

    private Long id;

    private String request;

    private String status;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String errorMessage;

    private Integer attempts;

    private LocalDateTime statusUpdateDate;

    public TaskDto(Long id, String request, StatusType status, String errorMessage,
                   Integer attempts, LocalDateTime statusUpdateDate) {
        this.id = id;
        this.request = request;
        this.status = status.toString();
        this.errorMessage = errorMessage;
        this.attempts = attempts;
        this.statusUpdateDate = statusUpdateDate;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getRequest() {
        return request;
    }

    public void setRequest(String request) {
        this.request = request;
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

    public Integer getAttempts() {
        return attempts;
    }

    public void setAttempts(Integer attempts) {
        this.attempts = attempts;
    }

    public LocalDateTime getStatusUpdateDate() {
        return statusUpdateDate;
    }

    public void setStatusUpdateDate(LocalDateTime statusUpdateDate) {
        this.statusUpdateDate = statusUpdateDate;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("id", id)
                .append("request", request)
                .append("status", status)
                .append("errorMessage", errorMessage)
                .append("attempts", attempts)
                .append("statusUpdateDate", statusUpdateDate)
                .toString();
    }
}
