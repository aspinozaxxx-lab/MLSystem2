package ru.skoltech.aeronetlab.urban.action.dataloader;

import com.amazonaws.services.s3.AmazonS3URI;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.locationtech.jts.geom.Geometry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSource;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.repository.business.RasterSourceRepository;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreator;
import ru.skoltech.aeronetlab.urban.service.queue.TaskMessage;
import ru.skoltech.aeronetlab.urban.service.util.GeometryUtil;
import ru.skoltech.aeronetlab.urban.service.workflow.TaskService;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class DataLoaderTaskCreator implements TaskCreator {

    @Autowired
    private RasterSourceRepository rasterSourceRepository;

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private TaskService taskService;

    @Autowired
    private ObjectMapper objectMapper;


    @Override
    public Set<Task> create(Stage stage) {
        return artifactRepository.findAllByWorkflowAndArtifactType(stage.getWorkflow(), ArtifactType.RAW_RASTER)
                .stream()
                .map(a -> createTaskForAreaOfInterest(stage, a))
                .collect(Collectors.toSet());
    }

    private Task createTaskForAreaOfInterest(Stage stage, Artifact artifact) {

        TaskMessage taskMessage = new TaskMessage();

        taskMessage.setInput(composeTaskInputs(stage, artifact.getAreaOfInterest()));

        AmazonS3URI targetUri = new AmazonS3URI(artifact.getUri());
        taskMessage.getOutput().put("bucket", targetUri.getBucket());
        taskMessage.getOutput().put("filename", targetUri.getKey());

        return taskService.create(stage, artifact.getAreaOfInterest(), taskMessage);
    }

    public Map<String, Object> composeTaskInputs(Stage stage, AreaOfInterest aoi) {
        Map<String, String> params = prepareStageParameters(stage, aoi, Set.of("buffer", "workers", "ignore_errors"), Set.of("header", "connection_limit"), Collections.emptySet(), Set.of());
        Map<String, Object> inputs = new HashMap<>();

        Workflow workflow = stage.getWorkflow();
        /* Get login credentials from select-source stage results */
        //TODO: Consider removing raster source and move credentials extraction logic here
        RasterSource rasterSource = rasterSourceRepository.findByWorkflow(workflow)
                .orElseThrow(() -> new RuntimeException("RasterSource for " + workflow + " doesn't exist."));

        rasterSource.getParams().entrySet().stream()
                .filter(e -> !(e.getKey().equals("login") || e.getKey().equals("password")))
                .forEach(e -> inputs.put(e.getKey(), e.getValue()));

        String login = rasterSource.getParams().get("login");
        String pass = rasterSource.getParams().get("password");
        if (login != null && pass != null) {
            List<String> credentials = Arrays.asList(login, pass);
            inputs.put("credentials", credentials);
        }

        /* Prepare AOI geometry */
        String buffer = params.get("buffer");
        Geometry geometry = aoi.getGeometry();
        if (buffer != null) {
            geometry = GeometryUtil.applyBufferParam(geometry, buffer);
        }
        inputs.put("aoi", geometry);

        String headerJson = params.get("header");
        if (headerJson != null) {
            TypeReference<HashMap<String, Object>> typeRef = new TypeReference<>() {};
            try {
                Map<String, Object> header = objectMapper.readValue(headerJson, typeRef);
                inputs.put("header", header);
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Unable to parse header: " + headerJson, e);
            }
        }

        Optional.ofNullable(params.get("connection_limit"))
                .ifPresent(p -> inputs.put("connection_limit", p));
        Optional.ofNullable(params.get("ignore_errors"))
                .ifPresent(p -> inputs.put("ignore_errors", p));

        return inputs;
    }
}
