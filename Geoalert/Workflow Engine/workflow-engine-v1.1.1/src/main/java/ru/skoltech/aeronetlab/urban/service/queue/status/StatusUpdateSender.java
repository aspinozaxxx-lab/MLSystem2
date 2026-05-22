package ru.skoltech.aeronetlab.urban.service.queue.status;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.core.TopicExchange;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

@Service
public class StatusUpdateSender {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private TopicExchange exchange;

    @Value("${rabbitmq.max.priority:10}")
    private int maxPriority;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    public void send(Workflow workflow) {
        log.debug("Send Status Update message for workflow " + workflow.getId());
        StatusUpdateMessage statusUpdateMessage = new StatusUpdateMessage(workflow.getId());

        MessageProperties properties = new MessageProperties();
        String priority = workflow.getParams()
                .getOrDefault("priority", String.valueOf(maxPriority / 2));
        properties.setPriority(Integer.valueOf(priority));

        Message message = rabbitTemplate.getMessageConverter().toMessage(statusUpdateMessage, properties);

        rabbitTemplate.send(exchange.getName(), "status-update", message);
    }
}
