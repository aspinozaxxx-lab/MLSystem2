package ru.skoltech.aeronetlab.markupstorage.queue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.core.TopicExchange;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class MessageSender {

    @Value("${worker.name}")
    private String workerName;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private TopicExchange exchange;

    private Logger log = LoggerFactory.getLogger(this.getClass());

    public void sendOk(String taskId) {
        Map<String, String> message = new HashMap<String, String>();
        message.put("task_id", taskId);
        message.put("status", "0");

        send(message);
    }

    public void sendFail(String taskId, String errorMessage) {
        Map<String, String> message = new HashMap<String, String>();
        message.put("task_id", taskId);
        message.put("status", "1");
        message.put("error_message", errorMessage);

        send(message);
    }

    private void send(Map<String, String> message) {
        log.info("Sending message to result queue; message=" + message);
        rabbitTemplate.convertAndSend(exchange.getName(), workerName, message);
    }
}
