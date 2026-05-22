package ru.skoltech.aeronetlab.urban.service.queue;


import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.converter.MappingJackson2MessageConverter;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.service.workflow.TaskStatusService;

import java.io.IOException;

@Service
public class MessageListener {

    @Autowired
    private MappingJackson2MessageConverter messageConverter;

    @Autowired
    private TaskStatusService taskStatusService;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @RabbitListener(queues = "#{resultQueues.getDeclarables()}")
    public void receiveMessage(Message message) {
        log.info(String.format("Received message in result queue: {%s}", message));

        ResponseMessage response = null;
        try {
            response = messageConverter.getObjectMapper()
                    .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
                    .readValue(message.getBody(), ResponseMessage.class);
            log.info(String.format("Parsed response message. {%s}", response));
        } catch (IOException e) {
            log.error("Error processing message in result queue: " + (new String(message.getBody())), e);
        }

        updateTaskStatus(response);
    }

    @Transactional
    protected void updateTaskStatus(ResponseMessage response) {
        try {
            taskStatusService.updateStatus(response);
        } catch (Exception e) {
            log.error("Couldn't update status of task_id=" + response.getTaskId() + " because of ", e);
        }
    }
}
