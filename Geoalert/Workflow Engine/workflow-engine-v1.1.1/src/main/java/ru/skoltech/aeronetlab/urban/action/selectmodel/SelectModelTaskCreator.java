package ru.skoltech.aeronetlab.urban.action.selectmodel;

import org.locationtech.jts.geom.Geometry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreator;
import ru.skoltech.aeronetlab.urban.service.queue.TaskMessage;
import ru.skoltech.aeronetlab.urban.service.workflow.TaskService;

import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class SelectModelTaskCreator implements TaskCreator {
    private static final Set<String> WD_PARAMS = Set.of("problem_name", "default_model");
    private static final Set<String> REQUIRED_PARAMS = Set.of("problem_name");

    @Autowired
    private TaskService taskService;

    @Autowired
    private ArtifactRepository artifactRepository;

    @Override
    public Set<Task> create(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        Set<Artifact> rawRasters = artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.RAW_RASTER);

        return rawRasters.stream().map(a -> createTask(stage, a)).collect(Collectors.toSet());
    }

    public Task createTask(Stage stage, Artifact artifact) {
        Map<String, String> params = prepareStageParameters(stage, artifact.getAreaOfInterest(), WD_PARAMS,
                Collections.emptySet(), Collections.emptySet(), REQUIRED_PARAMS);


        TaskMessage taskMessage = new TaskMessage();
        taskMessage.getInput().put("problem_name", params.get("problem_name"));
        taskMessage.getInput().put("default_model", params.get("default_model"));

        Geometry aoi = artifact.getAreaOfInterest().getGeometry();
        taskMessage.getInput().put("aoi", aoi);

        taskMessage.getInput().put("RAW_RASTER", artifact.getUri());

        return taskService.create(stage, artifact.getAreaOfInterest(), taskMessage);
    }
}
