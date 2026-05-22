package ru.skoltech.aeronetlab.urban.service.queue;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.core.TopicExchange;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import org.springframework.beans.factory.annotation.Value;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;

@Service
public class MessageSender {

  @Autowired
  private RabbitTemplate rabbitTemplate;

  @Autowired
  private TopicExchange exchange;

  @Autowired
  private ObjectMapper objectMapper;

  @Value("${rabbitmq.max.priority:10}")
  private int maxPriority;

  private final Logger log = LoggerFactory.getLogger(this.getClass());

  private final HashMap<String, Object> mockMapField = null;
  private final Type messageType;

  public MessageSender() {
    try {
      messageType = MessageSender.class.getDeclaredField("mockMapField").getGenericType();
    } catch (NoSuchFieldException e) {
      throw new RuntimeException(e);
    }
  }

  public void send(Task task) {
    TaskMessage taskMessage;

    try {
      taskMessage = objectMapper.readValue(task.getRequest(), TaskMessage.class);
    } catch (IOException e) {
      throw new RuntimeException("Error deserializing task message: " + task.getRequest(), e);
    }

    MessageProperties properties = new MessageProperties();
    int defaultPriority = maxPriority / 2;
    String priority = task.getStage().getWorkflow().getParams()
        .getOrDefault("priority", String.valueOf(defaultPriority));
    properties.setPriority(Integer.valueOf(priority));

    Message message = rabbitTemplate.getMessageConverter().toMessage(taskMessage, properties, messageType);

    Action action = task.getStage().getStageDefinition().getAction();
    String routingKey = action.getWorkerName()
        .orElseThrow(() -> new RuntimeException("Worker name not defined for " + action));

    log.info(String.format("Sending message with routingKey=%s; message={%s}; priority={%s}",
        routingKey,
        new String(message.getBody()),
        priority));

    rabbitTemplate.send(exchange.getName(), routingKey, message);
  }
}
