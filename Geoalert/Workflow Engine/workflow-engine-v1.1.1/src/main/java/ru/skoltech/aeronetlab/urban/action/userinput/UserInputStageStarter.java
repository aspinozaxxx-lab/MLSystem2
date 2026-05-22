package ru.skoltech.aeronetlab.urban.action.userinput;

import com.amazonaws.services.s3.AmazonS3URI;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Charsets;
import io.minio.MinioClient;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.StageStarter;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class UserInputStageStarter implements StageStarter {

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private JavaMailSender emailSender;

    @Value("${minio.location:http://localhost:9000}")
    private String minioLocation;

    @Value("${external.url:http://localhost:8060/}")
    private String externalUrl;

    @Autowired
    private MinioClient minioClient;

    @Autowired
    private ObjectMapper objectMapper;

    @Override
    synchronized public void start(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        List<String> inputs = stage.getStageDefinition().getParamAsList("inputs").stream()
                .map(s -> s.replace(" ", "-").replace("/", "-"))
                .collect(Collectors.toList());

        Set<String> requiredParams = new HashSet<>(stage.getStageDefinition().getParamAsList("required_metadata_params"));

        Map<String, Object> metadata = workflow.getParams().entrySet()
                .stream()
                .filter(entity -> requiredParams.contains(entity.getKey()))
                .collect(Collectors.toMap(Map.Entry<String, String>::getKey, entry -> stringToNumberSafe(entry.getValue())));

        String dir = "processing-" + workflow.getProcessingId();
        String bucket = stage.getStageDefinition().getParams().getOrDefault("bucket", "workflow");
        String uriPrefix = "s3://" + bucket + "/" + dir;

        String linkToArtifacts = parseUri(minioLocation).resolve("/" + bucket + "/" + dir + "/").toString();

        boolean allMetadataIsDefined = metadata.keySet().equals(requiredParams);

        if (allMetadataIsDefined) {
            //Skip all metadata except the input
            inputs = inputs.stream()
                    .filter(str -> str.equals("meta.geojson"))
                    .collect(Collectors.toList());
        }

        List<Artifact> artifacts = inputs.stream()
                .map(s -> new Artifact(workflow, ArtifactType.USER_INPUT, uriPrefix + "/" + s))
                .collect(Collectors.toList());

        artifactRepository.deleteAll(artifactRepository.findAllByWorkflowAndArtifactType(stage.getWorkflow(), ArtifactType.USER_INPUT));
        artifactRepository.saveAll(artifacts);

        boolean processing_already_started = UserInputStageStatusResolver.sentEmails.containsKey(linkToArtifacts);
        boolean send_once = Boolean.parseBoolean(stage.getStageDefinition().getParams().getOrDefault("send_once", "true"));
        if (processing_already_started && send_once) {
            return;
        }

        if (!metadata.isEmpty()) {
            allMetadataIsDefined = allMetadataIsDefined && saveMetadataToS3(uriPrefix + "/meta.geojson", metadata);
        }

        if (!allMetadataIsDefined) {
            sendMessage(stage, artifacts, linkToArtifacts);
        }
        if (!processing_already_started) {
            // We do not register every email in the statusResolver, only the first one
            UserInputStageStatusResolver.sentEmails.put(linkToArtifacts, System.currentTimeMillis());
        }
    }

    private static Object stringToNumberSafe(String str) {
        try {
            return Double.valueOf(str);
        } catch (NumberFormatException e) {
            return str;
        }
    }

    private void sendMessage(Stage stage, List<Artifact> artifacts, String linkToArtifacts) {
        Workflow workflow = stage.getWorkflow();

        String rasterUri = artifactRepository.findByWorkflowAndArtifactType(workflow, ArtifactType.RAW_RASTER)
                .map(Artifact::getUri)
                .orElseThrow(() -> new RuntimeException("RasterSource for " + workflow + " doesn't exist."));

        String rasterDirUri = rasterUri.substring(0, rasterUri.lastIndexOf('/') + 1);
        String linkToRaster = parseUri(minioLocation).resolve(rasterDirUri.replace("s3:/", "")).toString();

        Long wfId = stage.getWorkflow().getId();
        String linkToWf = parseUri(externalUrl).resolve("/api/v0/workflows/" + wfId).toString();
        String linkToSwagger =
                parseUri(externalUrl).resolve("/swagger-ui.html#/workflow-controller/getWorkflowUsingGET").toString();

        StringBuilder body = new StringBuilder();
        body.append("User input required for workflow ");
        body.append(workflow.getId());
        body.append(". Please ").append(htmlLink("upload", linkToArtifacts)).append(" the following artifacts:<br /><br />");
        artifacts.forEach(a -> body.append(a.getUri()).append("<br />"));
        body.append("<br /><br />Raster data is located at ").append(htmlLink(rasterUri, linkToRaster)).append(".");
        if (workflow.getParams().containsKey("url")) {
            body.append("<br />Raster Source URL: ").append(workflow.getParams().get("url"));
        }
        body.append("<br /><br />").append(htmlLink("Watch workflow " + wfId, linkToWf));
        body.append(" or use ").append(htmlLink("swagger", linkToSwagger)).append(".");

        Object[] recipientsObj = stage.getStageDefinition().getParamAsList("recipients").toArray();
        String[] recipients = Arrays.copyOf(recipientsObj, recipientsObj.length, String[].class);

        try {
            MimeMessage message = emailSender.createMimeMessage();

            MimeMessageHelper helper = new MimeMessageHelper(message, true);

            helper.setTo(recipients);
            helper.setSubject("User input for workflow " + workflow.getId());
            helper.setText(body.toString(), true);

            String recipientsStr = String.join(", ", recipients);
            log.info("Sending user-input message to " + recipientsStr);

            emailSender.send(message);
        } catch (MessagingException e) {
            throw new RuntimeException("Error trying to send email: " + e, e);
        }
    }

    private boolean saveMetadataToS3(String uri, Map<String, Object> metadata) {
        try {
            AmazonS3URI s3Uri = new AmazonS3URI(uri);

            InputStream tmplIs = UserInputStageStarter.class.getResourceAsStream("meta.geojson.template");
            assert tmplIs != null;
            String tmpl = new String(tmplIs.readAllBytes(), Charsets.UTF_8);
            String json = tmpl.replace("%METADATA%", objectMapper.writeValueAsString(metadata));

            log.debug("Saving metadata file: " + json + " to " + uri);
            InputStream is = new ByteArrayInputStream(json.getBytes(StandardCharsets.UTF_8));

            minioClient.putObject(s3Uri.getBucket(), s3Uri.getKey(), is, "application/json");
            return true;
        } catch (Exception e) {
            log.error("Unable to save Metadata to S3", e);
            return false;
        }
    }

    private String htmlLink(String caption, String uri) {
        return "<a href='" + uri + "'>" + caption + "</a>";
    }

    private URI parseUri(String uri) {
        try {
            return new URI(uri);
        } catch (URISyntaxException e) {
            throw new RuntimeException("Invalid uri: " + uri, e);
        }
    }
}
