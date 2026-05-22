package ru.skoltech.aeronetlab.urban;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.minio.MinioClient;
import io.minio.errors.InvalidEndpointException;
import io.minio.errors.InvalidPortException;
import org.jetbrains.annotations.NotNull;
import org.locationtech.spatial4j.io.jackson.ShapesAsGeoJSONModule;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.net.MalformedURLException;
import java.net.URL;

@SpringBootApplication
@EnableScheduling
@EnableAsync
public class Application {

	public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
	}

	@Autowired
	public void configureJackson(ObjectMapper objectMapper) {
	    objectMapper.registerModule(new ShapesAsGeoJSONModule());
    }

	@Value("${cors.allowed.origins:http://localhost:3000}")
	private String[] corsAllowedOrigins = new String[]{"http://localhost:3000"};

	@Bean
	public WebMvcConfigurer corsConfigurer() {
		return new WebMvcConfigurer() {
			@Override
			public void addCorsMappings(@NotNull CorsRegistry registry) {
				registry.addMapping("/api/**")
						.exposedHeaders("content-length", "content-disposition", "date", "accept-ranges")
						.allowedMethods("GET", "POST", "DELETE", "PUT", "OPTIONS", "PATCH")
						.allowedOrigins("*");
			}
		};
	}

	@Bean(name = "minioClient")
	public MinioClient getS3ClientInternally(@Value("${minio.access.key:mysupersecretkey}") String s3AccessKey,
								   @Value("${minio.secret.key:}") String s3SecretKey,
								   @Value("${minio.host:localhost}") String s3Host,
								   @Value("${minio.port:9000}") String s3Port) {
		return getS3Client(s3AccessKey, s3SecretKey, s3Host, Integer.parseInt(s3Port), false);
	}

	@Bean(name = "minioClientExternally")
	public MinioClient getS3ClientExternally(@Value("${minio.access.key:mysupersecretkey}") String s3AccessKey,
								   @Value("${minio.secret.key:}") String s3SecretKey,
								   @Value("${minio.location:http://localhost:9000}") String s3location) {
		URL s3Url;
		try {
			s3Url = new URL(s3location);
		} catch (MalformedURLException e) {
			throw new RuntimeException("Bad external minio endpoint", e);
		}

		int port = s3Url.getPort();
		if (port == -1) port = s3Url.getProtocol().equalsIgnoreCase("https") ? 443: 80;

		return getS3Client(
				s3AccessKey, s3SecretKey, s3Url.getHost(), port,
				s3Url.getProtocol().equalsIgnoreCase("https")
		);
	}

	private MinioClient getS3Client(String s3AccessKey, String s3SecretKey, String s3Host, int s3Port, boolean secure) {
		try {
			return new MinioClient(s3Host, s3Port, s3AccessKey, s3SecretKey, secure);
		} catch (InvalidEndpointException | InvalidPortException e) {
			throw new RuntimeException("Couldn't construct minio client", e);
		}
	}
}