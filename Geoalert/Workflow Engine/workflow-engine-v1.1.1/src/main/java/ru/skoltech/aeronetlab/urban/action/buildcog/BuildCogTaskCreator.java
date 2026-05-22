package ru.skoltech.aeronetlab.urban.action.buildcog;

import org.locationtech.jts.geom.Geometry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.business.RasterLayer;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreator;
import ru.skoltech.aeronetlab.urban.service.queue.TaskMessage;
import ru.skoltech.aeronetlab.urban.service.util.GeometryUtil;
import ru.skoltech.aeronetlab.urban.service.workflow.TaskService;

import java.util.Collections;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class BuildCogTaskCreator implements TaskCreator {
    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private TaskService taskService;

    @Override
    public Set<Task> create(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        RasterLayer rasterLayer = Optional.ofNullable(workflow.getRasterLayer())
                .orElseThrow(() -> new RuntimeException("Target RasterLayer is not defined for " + workflow));

        Set<Artifact> rawRasters = artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.RAW_RASTER);

        if (rawRasters.isEmpty()) {
            log.warn("Cannot build COG. No rasters were found for workflow " + workflow.getId() + " (processing " + workflow.getProcessingId() + ")");
            return Collections.emptySet();
        }

        return rawRasters.stream()
                .map(a -> createTask(stage, a, rasterLayer.getUri()))
                .collect(Collectors.toSet());
    }

    private Task createTask(Stage stage, Artifact artifact, String layerUri) {
        TaskMessage taskMessage = new TaskMessage();
        taskMessage.getInput().put("task", "build_cog");
        taskMessage.getInput().put("raster_source", artifact.getUri());

        String channels = stage.getStageDefinition().getParams().get("channels");
        if (StringUtils.hasLength(channels)) taskMessage.getInput().put("channels", channels);

        String cropToMask = stage.getStageDefinition().getParams().getOrDefault("crop_to_mask", "true");
        if (Boolean.parseBoolean(cropToMask)) {
            Geometry mask = artifact.getAreaOfInterest().getGeometry();
            String buffer = stage.getStageDefinition().getParams().get("buffer");
            if (buffer != null) {
                mask = GeometryUtil.applyBufferParam(mask, buffer);
            }

            //TODO: 'mask' parameter is obsolete and will be removed after cog-builder was updated
            taskMessage.getInput().put("mask", mask);
            taskMessage.getInput().put("aoi", mask);
        }

        taskMessage.getOutput().put("target_uri", layerUri + "/area-" + artifact.getId() + ".tif");

        return taskService.create(stage, artifact.getAreaOfInterest(), taskMessage);
    }
}
