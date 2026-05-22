package ru.skoltech.aeronetlab.urban.action.importvector;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.VectorLayer;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.VectorLayerRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowRepository;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.TaskStageStarter;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;

@Service
public class ImportVectorStageStarter extends TaskStageStarter {

    @Autowired
    private VectorLayerRepository vectorLayerRepository;

    @Autowired
    private WorkflowRepository workflowRepository;

    @Override
    public void start(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        Map<String, String> stageParams = stage.getStageDefinition().getParams();
        Map<String, String> workflowParams = workflow.getParams();
        Optional<UUID> predefinedId = Optional
                .ofNullable(workflowParams.getOrDefault("vector-layer-id", stageParams.get("vector-layer-id")))
                .map(UUID::fromString);
        UUID layerId = predefinedId.orElseGet(UUID::randomUUID);

        VectorLayer vectorLayer = vectorLayerRepository.findByLayerId(layerId)
                .orElseGet(() -> createVectorLayer(layerId));

        workflow.setVectorLayer(vectorLayer);
        workflowRepository.save(workflow);

        super.start(stage);
    }

    private VectorLayer createVectorLayer(UUID layerId) {
        VectorLayer vectorLayer = new VectorLayer();
        vectorLayer.setLayerId(layerId);
        return vectorLayerRepository.save(vectorLayer);
    }
}
