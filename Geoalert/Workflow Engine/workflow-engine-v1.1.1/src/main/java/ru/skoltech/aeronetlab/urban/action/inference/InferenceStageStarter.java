package ru.skoltech.aeronetlab.urban.action.inference;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.TaskStageStarter;

import java.util.UUID;

@Service
public class InferenceStageStarter extends TaskStageStarter {

    @Autowired
    private ArtifactRepository artifactRepository;

    @Override
    public void start(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        artifactRepository.deleteAll(
                artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.RAW_VECTOR)
        );

        String bucket = stage.getStageDefinition().getParams().getOrDefault("bucket", "workflow");
        String dir = String.join("/", "workflow-" + workflow.getId(), UUID.randomUUID().toString());
        String targetUriPrefix = "s3://" + bucket + "/" + dir;

        artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.RAW_RASTER)
                .forEach(a -> createVectorArtifact(a, targetUriPrefix));

        super.start(stage);
    }

    private void createVectorArtifact(Artifact rasterArtifact, String targetUriPrefix) {
        String uri = targetUriPrefix + "/area-" + rasterArtifact.getAreaOfInterest().getId() + ".geojson";
        Artifact vectorArtifact = new Artifact(rasterArtifact.getWorkflow(), rasterArtifact.getAreaOfInterest(), ArtifactType.RAW_VECTOR, uri);

        artifactRepository.save(vectorArtifact);
    }
}
