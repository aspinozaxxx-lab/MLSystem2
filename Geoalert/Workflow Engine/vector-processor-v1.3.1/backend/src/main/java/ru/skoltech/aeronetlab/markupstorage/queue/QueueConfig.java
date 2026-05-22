package ru.skoltech.aeronetlab.markupstorage.queue;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.TopicExchange;
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

import java.util.HashMap;
import java.util.Map;

@Configuration
public class QueueConfig implements RabbitListenerConfigurer {

    private String url;
    private Integer port;
    private String user;
    private String password;
    private String taskQueuePostfix;
    private String resultQueuePostfix;

    public QueueConfig(@Value("${rabbitmq.host}") String url,
                       @Value("${rabbitmq.node.port}") Integer port,
                       @Value("${rabbitmq.default.user}") String user,
                       @Value("${rabbitmq.default.pass}") String password,
                       @Value("${input.queue}") String taskQueuePostfix,
                       @Value("${output.queue}") String resultQueuePostfix) {
        this.url = url;
        this.port = port;
        this.user = user;
        this.password = 
        this.taskQueuePostfix = taskQueuePostfix;
        this.resultQueuePostfix = resultQueuePostfix;
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
    public TopicExchange getTopicExchange(@Value("${worker.name}") String workerName) {
        return new TopicExchange(workerName + ".exchange");
    }

    @Bean
    public Jackson2JsonMessageConverter producerJackson2MessageConverter(ObjectMapper objectMapper) {
        return new Jackson2JsonMessageConverter(objectMapper);
    }

    @Bean
    public MappingJackson2MessageConverter consumerJackson2MessageConverter() {
        MappingJackson2MessageConverter converter = new MappingJackson2MessageConverter();
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

    @Bean(name = "taskQueue")
    public Queue getTaskQueue(@Value("${worker.name}") String workerName,
                              @Value("${input.queue}") String taskQueuePostfix,
                              @Value("${rabbitmq.max.priority:10}") Integer maxPriority) {
        Map<String, Object> args = new HashMap<>();
        args.put("x-max-priority", maxPriority);

        return new Queue(workerName + taskQueuePostfix, true, false, false, args);
    }

    @Bean(name = "resultQueue")
    public Queue getResultQueue(@Value("${worker.name}") String workerName,
                                @Value("${output.queue}") String resultQueuePostfix) {
        return new Queue(workerName + resultQueuePostfix);
    }

    @Bean(name = "queueBinding")
    public Binding getBinding(@Value("${worker.name}") String workerName,
                              TopicExchange exchange, Queue resultQueue) {
        return BindingBuilder.bind(resultQueue).to(exchange).with(workerName);
    }
}
