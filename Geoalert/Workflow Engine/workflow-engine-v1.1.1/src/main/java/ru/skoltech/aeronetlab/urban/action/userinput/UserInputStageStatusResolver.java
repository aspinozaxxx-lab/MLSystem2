package ru.skoltech.aeronetlab.urban.action.userinput;

import com.amazonaws.services.s3.AmazonS3URI;
import io.minio.MinioClient;
import io.minio.Result;
import io.minio.messages.Item;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Message;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.service.action.statusresolver.StageStatusResolver;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

@Service
public class UserInputStageStatusResolver implements StageStatusResolver {

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Value("#{systemEnvironment['USER_INPUT_TIMEOUT_HOURS'] ?: '72'}")
    private  int userInputTimeoutHours = 72;

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private MinioClient minioClient;

    public static final Map<String, Long> sentEmails = new ConcurrentHashMap<>();

    @Override
    public void resolve(StageStatus stageStatus) {
        Workflow workflow = stageStatus.getStage().getWorkflow();

        Set<Artifact> artifacts =
                artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.USER_INPUT);

        boolean isOk = artifacts.stream()
                .map(Artifact::getUri)
                .collect(Collectors.groupingBy(s -> s.substring(0, s.lastIndexOf('/'))))
                .entrySet()
                .stream()
                .allMatch(e -> checkDir(e.getKey() + "/", e.getValue()));

        LocalDateTime startDate = stageStatus.getStartDate();
        if (startDate == null) {
            startDate = workflow.getCreateDate();
        }
        if (isOk) {
            stageStatus.setStatus(StatusType.OK);
            sentEmails.remove(workflow.getProcessingId());
        } else if (startDate.plus(userInputTimeoutHours, ChronoUnit.HOURS).isAfter(LocalDateTime.now())) {
            stageStatus.setStatus(StatusType.IN_PROGRESS);
        } else {
            String message = "Workflow " + workflow.getId() + " for processing " + workflow.getProcessingId() +
                    " was failed because no user input was provided after " + userInputTimeoutHours + " hours";
            log.warn(message);
            stageStatus.setStatus(StatusType.FAILED);
            stageStatus.getMessages().add(new Message("workflow_engine.userInputTimeout", Collections.emptyMap(), message));
            sentEmails.remove(workflow.getProcessingId());
        }
    }

    private boolean checkDir(String dir, List<String> uris) {
        String urisStr = String.join((", "), uris);
        log.debug("Checking if " + urisStr + " are present in " + dir);

        Set<String> contents;
        try {
            contents = listObjectsInDir(dir);
        } catch (RuntimeException e) {
            log.error("Error listing contents of " + dir, e);
            return false;
        }

        log.debug("Artifacts present in " + dir + ": " + String.join((", "), contents));

        return uris.stream()
                .map(uri -> uri.substring(uri.lastIndexOf('/') + 1))
                .allMatch(contents::contains);
    }

    private Set<String> listObjectsInDir(String dir) {
        AmazonS3URI s3Uri = new AmazonS3URI(dir);
        return StreamSupport.stream(listObjects(s3Uri).spliterator(), false)
        .map(r -> {
            try {
                return r.get().objectName();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        })
        .map(s -> s.substring(s.lastIndexOf('/') + 1))
        .collect(Collectors.toSet());
    }

    private Iterable<Result<Item>> listObjects(AmazonS3URI s3Uri) {
        return minioClient.listObjects(s3Uri.getBucket(), s3Uri.getKey(), true);
    }
}