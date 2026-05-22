package ru.skoltech.aeronetlab.urban.dto.workflow;

import org.apache.commons.lang3.builder.ToStringBuilder;

public class TaskIdDto {

    private Long taskId;

    private Long stageId;

    public TaskIdDto() {}

    public TaskIdDto(Long taskId, Long stageId) {
        this.taskId = taskId;
        this.stageId = stageId;
    }

    public Long getTaskId() {
        return taskId;
    }

    public void setTaskId(Long taskId) {
        this.taskId = taskId;
    }

    public Long getStageId() {
        return stageId;
    }

    public void setStageId(Long stageId) {
        this.stageId = stageId;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("taskId", taskId)
                .append("stageId", stageId)
                .toString();
    }
}
