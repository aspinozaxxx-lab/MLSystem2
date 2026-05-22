package ru.skoltech.aeronetlab.markupstorage;

import com.bedatadriven.jackson.datatype.jts.JtsModule;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.minio.MinioClient;
import io.minio.errors.InvalidEndpointException;
import io.minio.errors.InvalidPortException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

@SpringBootApplication
@EntityScan(basePackages = "ru.skoltech.aeronetlab.markupstorage.dao")
@EnableAsync
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @Autowired
    public void configureJackson(ObjectMapper objectMapper) {
        objectMapper.registerModule(new JtsModule());
    }

    @Bean
    public MinioClient getS3Client(@Value("${minio.access.key}") String s3AccessKey,
                                   @Value("${minio.secret.key}") String s3SecretKey,
                                   @Value("${minio.host}") String s3Host,
                                   @Value("${minio.port}") String s3Port) {
        try {
            return new MinioClient(s3Host, Integer.valueOf(s3Port), s3AccessKey, s3SecretKey, false);
        } catch (InvalidEndpointException | InvalidPortException e) {
            throw new RuntimeException("Couldn't construct minio client: " + e.toString(), e);
        }
    }

    @Bean(name = "taskExecutor")
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(16);
        executor.setMaxPoolSize(256);
        return executor;
    }
}
