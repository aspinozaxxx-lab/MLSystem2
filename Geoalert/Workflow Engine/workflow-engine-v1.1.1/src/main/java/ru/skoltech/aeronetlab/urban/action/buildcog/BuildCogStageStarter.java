package ru.skoltech.aeronetlab.urban.action.buildcog;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.RasterLayer;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.RasterLayerRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowRepository;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.TaskStageStarter;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;

@Service
public class BuildCogStageStarter extends TaskStageStarter {

    @Autowired
    private RasterLayerRepository rasterLayerRepository;

    @Autowired
    private WorkflowRepository workflowRepository;

    @Override
    public void start(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        Map<String, String> stageParams = stage.getStageDefinition().getParams();
        Map<String, String> workflowParams = workflow.getParams();
        Optional<String> predefinedUri = Optional
                .ofNullable(workflowParams.getOrDefault("raster-layer-uri", stageParams.get("raster-layer-uri")));

        String uri = predefinedUri
                .orElseGet(() -> String.join(
                        "/",
                        "s3://" + stageParams.getOrDefault("bucket", "workflow"),
                        "workflow-" + workflow.getId(),
                        UUID.randomUUID().toString()
                ));

        RasterLayer rasterLayer = rasterLayerRepository.findByUri(uri).orElseGet(() -> createRasterLayer(uri));

        workflow.setRasterLayer(rasterLayer);
        workflowRepository.save(workflow);

        super.start(stage);
    }

    private RasterLayer createRasterLayer(String uri) {
        RasterLayer rasterLayer = new RasterLayer();
        rasterLayer.setUri(uri);

        return rasterLayerRepository.save(rasterLayer);
    }
}
