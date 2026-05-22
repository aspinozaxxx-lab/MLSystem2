package ru.skoltech.aeronetlab.urban.action.inference;

import com.amazonaws.services.s3.AmazonS3URI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.definition.BlockConfig;
import ru.skoltech.aeronetlab.urban.entity.workflow.BlockParameters;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreator;
import ru.skoltech.aeronetlab.urban.service.queue.TaskMessage;
import ru.skoltech.aeronetlab.urban.service.util.GeometryUtil;
import ru.skoltech.aeronetlab.urban.service.workflow.TaskService;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class InferenceTaskCreator implements TaskCreator {
    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private TaskService taskService;

    @Override
    public Set<Task> create(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        Set<Artifact> userInputs =
                artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.USER_INPUT);

        return artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.RAW_RASTER)
                .stream()
                .map(a -> createTask(stage, a, userInputs))
                .collect(Collectors.toSet());
    }

    @Override
    public Map<String, String> prepareStageParameters(Stage stage,
                                                      AreaOfInterest aoi,
                                                      Set<String> wdOnlyParamNames,
                                                      Set<String> workflowParamNames,
                                                      Set<String> aoiParamNames,
                                                      Set<String> requiredParamNames) {
        Set<String> actualWdParamNames = stage.getStageDefinition().getParams().keySet();
        Map<String, String> params = TaskCreator.super.prepareStageParameters(
                stage,
                aoi,
                actualWdParamNames,
                Set.of("model"),
                Set.of("model"),
                Collections.emptySet()
        );

        String model = aoi.getParams().get("model");

        Set<String> keysToRemove = params.keySet()
                .stream()
                .filter(k -> k.startsWith("pipeline."))
                .collect(Collectors.toSet());

        if (model != null && params.containsKey("pipeline." + model)) {
            log.info("Used 'pipeline." + model + "' as a 'pipeline' parameter for Inference stage");
            params.put("pipeline", params.get("pipeline." + model));
        }

        keysToRemove.forEach(params::remove);

        if (!params.containsKey("pipeline")) {
            throw new RuntimeException("Inference stage definition must define 'pipeline'  parameter");
        }

        return params;
    }

    private Task createTask(Stage stage, Artifact rasterArtifact, Set<Artifact> userInputs) {
        AreaOfInterest aoi = rasterArtifact.getAreaOfInterest();
        Map<String, String> params = prepareStageParameters(stage, aoi,
                Collections.emptySet(), Collections.emptySet(), Collections.emptySet(), Collections.emptySet());

        Optional<Artifact> vectorArtifact = artifactRepository.findByAreaOfInterestAndArtifactType(
                rasterArtifact.getAreaOfInterest(), ArtifactType.RAW_VECTOR
        );

        TaskMessage taskMessage = new TaskMessage();

        Map<String, Object> inputs = taskMessage.getInput();
        inputs.putAll(params);

        Map<String, String> stageParams = stage.getStageDefinition().getParams();
        Map<String, String> workflowParams = stage.getWorkflow().getParams();

        String buffer = stageParams.get("buffer");
        if (buffer != null) {
            inputs.put("aoi", GeometryUtil.applyBufferParam(aoi.getGeometry(), buffer));
        } else {
            inputs.put("aoi", aoi.getGeometry());
        }

        Collection<BlockConfig> blockConfigs = stage
                .getWorkflow()
                .getWorkflowDefinitionVer()
                .getBlockConfig();
        if (blockConfigs != null) {
            Collection<BlockParameters> blockParams = stage.getWorkflow().getBlockParams();
            if (blockParams == null) {
                blockParams = Collections.emptyList();
            }
            Map <String, Boolean> blockInputs = new HashMap<>();
            for (BlockConfig bc : blockConfigs) {
                Optional<BlockParameters> opt = blockParams
                        .stream()
                        .filter(bp -> bc.getName().equals(bp.getName()))
                        .findFirst();
                boolean enabled = opt.map(BlockParameters::isEnabled).orElseGet(() -> !bc.isOptional());
                blockInputs.put(bc.getName(), enabled);
            }
            inputs.put("blocks", blockInputs);
        }

        List<Map<String, String>> sourceData = new ArrayList<>();
        Map<String, String> raster = new HashMap<>();
        raster.put("name", "input.tif");
        raster.put("path", rasterArtifact.getUri());
        sourceData.add(raster);
        userInputs.forEach(a -> sourceData.add(buildUserInput(a)));
        taskMessage.getInput().put("source_data", sourceData);

        List<Map<String, String>> outputData = new ArrayList<>();

        if (vectorArtifact.isPresent()) {
            AmazonS3URI vectorUri = new AmazonS3URI(vectorArtifact.get().getUri());
            Map<String, String> vector = new HashMap<>();
            vector.put("name", "output.geojson");
            vector.put("path", vectorArtifact.get().getUri());
            outputData.add(vector);
        }

        Optional<String> outputBinary = Optional
                .ofNullable(workflowParams.getOrDefault("binary-output-uri", stageParams.get("binary-output-uri")));

        if (outputBinary.isPresent()) {
            Artifact binaryArtifact = createBinaryArtifact(aoi, outputBinary.get());
            Map<String, String> binary = new HashMap<>();
            binary.put("name", "output.tar");
            binary.put("path", binaryArtifact.getUri());
            outputData.add(binary);
        }

        taskMessage.getOutput().put("output_data", outputData);
        return taskService.create(stage, rasterArtifact.getAreaOfInterest(), taskMessage);
    }

    private Artifact createBinaryArtifact(AreaOfInterest aoi, String uri) {
        Artifact artifact = new Artifact(aoi.getWorkflow(), aoi, ArtifactType.BINARY_OUTPUT, uri);

        return artifactRepository.save(artifact);
    }

    private Map<String, String> buildUserInput(Artifact artifact) {
        Map<String, String> input = new HashMap<>();
        String uri = artifact.getUri();
        input.put("name", uri.substring(uri.lastIndexOf('/') + 1));
        input.put("path", uri);
        return input;
    }
}
