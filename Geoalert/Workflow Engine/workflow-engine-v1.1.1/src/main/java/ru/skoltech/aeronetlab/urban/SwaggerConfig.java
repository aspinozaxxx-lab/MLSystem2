package ru.skoltech.aeronetlab.urban;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SwaggerConfig {

//    @Bean
//    public GroupedOpenApi markupStorageApi() {
//        return GroupedOpenApi.builder()
//                .group("urban-api")
//                .displayName("Workflow Engine API")
//                .pathsToMatch("/api/v0.*")
//                .build();
//    }

    @Bean
    public OpenAPI weAPI() {
        return new OpenAPI()
                .info(new Info().title("Workflow Engine API")
                        .description("Workflow Engine internal API")
                        .version("v0")
                );
    }

}
