package ru.skoltech.aeronetlab.urban.service.queue.status;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.converter.MappingJackson2MessageConverter;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.service.workflow.StatusUpdater;

import java.io.IOException;

@Service
public class StatusUpdateListener {

    @Autowired
    private MappingJackson2MessageConverter messageConverter;

    @Autowired
    private StatusUpdater statusUpdater;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @RabbitListener(queues = "status-update")
    public void receiveMessage(Message message) {
        log.info(String.format("Received message in status-update queue: {%s}", message));

        StatusUpdateMessage statusUpdateMessage = null;
        try {
            statusUpdateMessage = messageConverter.getObjectMapper()
                    .setPropertyNamingStrategy( PropertyNamingStrategies.LOWER_CAMEL_CASE)
                    .readValue(message.getBody(), StatusUpdateMessage.class);
        } catch (IOException e) {
            log.error("Error processing status update message", e);
        }

        updateTaskStatus(statusUpdateMessage);
    }

    private void updateTaskStatus(StatusUpdateMessage statusUpdateMessage) {
        try {
            statusUpdater.updateWorkflow(statusUpdateMessage.getWorkflowId());
        } catch (RuntimeException e) {
            log.error("Couldn't update status of workflowId=" + statusUpdateMessage.getWorkflowId(), e);
        }
    }
}
