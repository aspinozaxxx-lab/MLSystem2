package ru.skoltech.aeronetlab.urban.action.importvector;

import com.amazonaws.services.s3.AmazonS3URI;
import org.locationtech.jts.geom.Geometry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.*;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.AreaOfInterestRepository;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreator;
import ru.skoltech.aeronetlab.urban.service.queue.TaskMessage;
import ru.skoltech.aeronetlab.urban.service.util.GeometryUtil;
import ru.skoltech.aeronetlab.urban.service.workflow.TaskService;

import java.util.*;

@Service
public class ImportVectorTaskCreator implements TaskCreator {

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private TaskService taskService;

    @Autowired
    private AreaOfInterestRepository areaOfInterestRepository;

    @Override
    public Set<Task> create(Stage stage) {

        Artifact artifact =
                artifactRepository.findByWorkflowAndArtifactType(stage.getWorkflow(), ArtifactType.SIMPLIFIED_GEOJSON)
                        .orElseGet(() ->
                                artifactRepository.findByWorkflowAndArtifactType(stage.getWorkflow(), ArtifactType.RAW_VECTOR)
                                        .orElseThrow(() -> new RuntimeException("Couldn't find input artifact for " + stage))
                        );
        return new HashSet<>(List.of(createTask(stage, artifact)));
    }

    private Task createTask(Stage stage, Artifact artifact) {
        TaskMessage taskMessage = new TaskMessage();

        Workflow workflow = stage.getWorkflow();

        VectorLayer vectorLayer = Optional.ofNullable(workflow.getVectorLayer())
                .orElseThrow(() -> new RuntimeException("Target VectorLayer for " + workflow + " is not defined."));

        taskMessage.getInput().put("task", "import_vector");
        AmazonS3URI s3Uri = new AmazonS3URI(artifact.getUri());
        taskMessage.getInput().put("bucket", s3Uri.getBucket());
        taskMessage.getInput().put("filename", s3Uri.getKey());

        Geometry mask = areaOfInterestRepository.findByWorkflow(stage.getWorkflow()).orElseThrow().getGeometry();
        String buffer = stage.getStageDefinition().getParams().get("buffer");
        if (buffer != null) mask = GeometryUtil.applyBufferParam(mask, buffer);
        taskMessage.getInput().put("mask", mask);

        boolean cropToMask = Optional.ofNullable(stage.getStageDefinition().getParams().get("crop_to_mask"))
                .filter(s -> s.equalsIgnoreCase("true"))
                .isPresent();
        taskMessage.getInput().put("crop_to_mask", cropToMask);

        String mergeStrategy = Optional.ofNullable(stage.getStageDefinition().getParams().get("merge_strategy"))
                .orElse("INSTANCE_SEGMENTATION");
        taskMessage.getInput().put("merge_strategy", mergeStrategy);

        List<String> keyProperties = stage.getStageDefinition().getParamAsList("key_properties");
        taskMessage.getInput().put("key_properties", keyProperties);

        taskMessage.getOutput().put("layer_id", vectorLayer.getLayerId());
        taskMessage.getOutput().put("layer_name", "workflow-" + workflow.getId());

        return taskService.create(stage, artifact.getAreaOfInterest(), taskMessage);
    }
}