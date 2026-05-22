package ru.skoltech.aeronetlab.urban.service.queue.status;

public class StatusUpdateMessage {

    private long workflowId;

    public StatusUpdateMessage() {}

    public StatusUpdateMessage(long workflowId) {
        this.workflowId = workflowId;
    }

    public long getWorkflowId() {
        return workflowId;
    }

    public void setWorkflowId(long workflowId) {
        this.workflowId = workflowId;
    }
}
