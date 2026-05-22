package ru.skoltech.aeronetlab.urban.service.queue;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.locationtech.spatial4j.io.jackson.ShapesAsGeoJSONModule;
import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.annotation.RabbitListenerConfigurer;
import org.springframework.amqp.rabbit.connection.CachingConnectionFactory;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.rabbit.listener.RabbitListenerEndpointRegistrar;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.converter.MappingJackson2MessageConverter;
import org.springframework.messaging.handler.annotation.support.DefaultMessageHandlerMethodFactory;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Configuration
public class QueueConfig implements RabbitListenerConfigurer {

    private final String url;
    private final Integer port;
    private final String user;
    private final String password;
    private final String taskQueuePostfix;
    private final String resultQueuePostfix;
    private final int maxPriority;

    public QueueConfig(@Value("${rabbitmq.host:localhost}") String url,
                       @Value("${rabbitmq.node.port:5672}") Integer port,
                       @Value("${rabbitmq.default.user:user}") String user,
                       @Value("${rabbitmq.default.pass:password}") String password,
                       @Value("${input.queue:.tasks.queue}") String taskQueuePostfix,
                       @Value("${output.queue:.result.queue}") String resultQueuePostfix,
                       @Value("${rabbitmq.max.priority:10}") int maxPriority) {
        this.url = url;
        this.port = port;
        this.user = user;
        this.password = 
        this.taskQueuePostfix = taskQueuePostfix;
        this.resultQueuePostfix = resultQueuePostfix;
        this.maxPriority = maxPriority;
    }

    @Bean
    public ConnectionFactory connectionFactory() {
        CachingConnectionFactory connectionFactory = new CachingConnectionFactory(url);
        connectionFactory.setUsername(user);
        connectionFactory.setPassword(password);
        connectionFactory.setPort(port);
        return connectionFactory;
    }

    @Bean(name = "exchange")
    public TopicExchange getTopicExchange() {
        return new TopicExchange("task.exchange");
    }

    @Bean
    public Jackson2JsonMessageConverter producerJackson2MessageConverter(ObjectMapper objectMapper) {
        return new Jackson2JsonMessageConverter(objectMapper);
    }

    @Bean
    public MappingJackson2MessageConverter consumerJackson2MessageConverter() {
        MappingJackson2MessageConverter converter = new MappingJackson2MessageConverter();
        converter.getObjectMapper().registerModule(new ShapesAsGeoJSONModule());
        return converter;
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory,
                                         Jackson2JsonMessageConverter jackson2MessageConverter) {
        RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
        rabbitTemplate.setMessageConverter(jackson2MessageConverter);
        rabbitTemplate.setChannelTransacted(true);
        return rabbitTemplate;
    }

    @Bean
    public DefaultMessageHandlerMethodFactory messageHandlerMethodFactory() {
        DefaultMessageHandlerMethodFactory factory = new DefaultMessageHandlerMethodFactory();
        factory.setMessageConverter(consumerJackson2MessageConverter());
        return factory;
    }

    @Override
    public void configureRabbitListeners(final RabbitListenerEndpointRegistrar registrar) {
        registrar.setMessageHandlerMethodFactory(messageHandlerMethodFactory());
    }

    @Bean(name = "taskQueues")
    public Declarables getTaskQueues() {
        Map<String, Object> args = new HashMap<>();
        args.put("x-max-priority", maxPriority);

        return new Declarables(Arrays.stream(Action.values())
                .filter(a -> a.getWorkerName().isPresent())
                .map(a -> a.getWorkerName().get())
                .map(w -> (Declarable) new Queue(w + taskQueuePostfix, true, false, false, args))
                .toArray(Declarable[]::new));
    }

    @Bean(name = "queueBindings")
    public Declarables getBindings(TopicExchange exchange, Declarables taskQueues) {
        return new Declarables(taskQueues.getDeclarables()
                .stream()
                .map(d -> (Queue) d)
                .map(q -> BindingBuilder.bind(q).to(exchange).with(q.getName().replace(taskQueuePostfix, "")))
                .toArray(Declarable[]::new));
    }

    @Bean(name = "resultQueues")
    public Declarables getResultQueues() {
        return new Declarables(Arrays.stream(Action.values())
                .filter(a -> a.getWorkerName().isPresent())
                .map(a -> a.getWorkerName().get())
                .map(w -> (Declarable) new Queue(w + resultQueuePostfix))
                .toArray(Declarable[]::new));
    }

    @Bean(name = "statusUpdateQueue")
    public Queue getStatusUpdateQueue() {
        Map<String, Object> args = new HashMap<>();
        args.put("x-max-priority", maxPriority);

        return new Queue("status-update", true, false, false, args);
    }

    @Bean(name = "statusUpdateBinding")
    public Binding getStatusUpdateBinding(TopicExchange exchange, Queue statusUpdateQueue) {
        return BindingBuilder.bind(statusUpdateQueue)
                .to(exchange)
                .with("status-update");
    }
}
