package ru.skoltech.aeronetlab.markupstorage.we;

import java.time.Duration;
import java.util.Optional;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.boot.web.client.RestTemplateBuilder;


@Service
public class RunCheck {

    private final RestTemplate restTemplate;

    public RunCheck(RestTemplateBuilder restTemplateBuilder) {
        this.restTemplate = restTemplateBuilder
                .setConnectTimeout(Duration.ofSeconds(5))
                .setReadTimeout(Duration.ofSeconds(10))
                .build();
    }

    public String checkTask(String url) {
      Optional<String> response = Optional.ofNullable(restTemplate.getForObject(url, String.class));
      
      String status = response.orElseThrow(() -> new IllegalStateException("Failed to parse runcheck response"));
      
      return status;
    }
}