package ru.skoltech.aeronetlab.urban.action.dataloader;

import com.amazonaws.services.s3.AmazonS3URI;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.minio.MinioClient;
import org.apache.commons.codec.digest.DigestUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSource;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSourceType;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.repository.business.RasterSourceRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.AreaOfInterestRepository;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.TaskStageStarter;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreators;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class DataLoaderStageStarter extends TaskStageStarter {

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private RasterSourceRepository rasterSourceRepository;

    @Autowired
    private AreaOfInterestRepository areaOfInterestRepository;

    @Autowired
    private TaskCreators taskCreators;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private MinioClient minioClient;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Override
    public void start(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        artifactRepository.deleteAll(
                artifactRepository.findAllByWorkflowAndArtifactType(workflow, ArtifactType.RAW_RASTER)
        );

        RasterSource rasterSource = rasterSourceRepository.findByWorkflow(workflow)
                .orElseThrow(() -> new RuntimeException("No RasterSource exists for " + workflow));

        if (rasterSource.getRasterSourceType() == RasterSourceType.LOCAL) {
            useLocalData(stage, rasterSource);
        } else {
            useDataloaderWorker(stage);
        }
    }

    private void useDataloaderWorker(Stage stage) {
        Workflow workflow = stage.getWorkflow();

        Map<String, String> stageParams = stage.getStageDefinition().getParams();

        AreaOfInterest aoi = areaOfInterestRepository.findByWorkflow(workflow)
                .orElseThrow(() -> new RuntimeException("No area of interest found for " + workflow));

        DataLoaderTaskCreator taskCreator = ((DataLoaderTaskCreator) taskCreators.get(Action.DATA_LOADER));
        Map<String, Object> inputs = taskCreator.composeTaskInputs(stage, aoi);
        String cacheKey;
        try {
            cacheKey = DigestUtils.md5Hex(objectMapper.writeValueAsString(inputs));
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Error serializing task message: " + e, e);
        }

        if (Boolean.parseBoolean(stageParams.getOrDefault("use_cache", "false"))) {
            log.debug("Looking for cached artifacts for " + stage + " with cacheKey=" + cacheKey);

            List<Artifact> oldArtifacts = artifactRepository.findTop5ByArtifactTypeAndCacheKeyOrderByIdDesc(
                    ArtifactType.RAW_RASTER, cacheKey
            );

            for (Artifact oldArtifact : oldArtifacts) {
                String uri = oldArtifact.getUri();
                AmazonS3URI s3Uri = new AmazonS3URI(uri);
                log.debug("Checking cached artifact for " + stage + ": " + uri);
                try {
                    if (minioClient.listObjects(s3Uri.getBucket(), s3Uri.getKey()).iterator().hasNext()) {
                        log.debug("Found cached artifact for " + stage + ": " + uri);
                        createRasterData(aoi, uri, cacheKey);
                        stage.getStageStatus().setStatus(StatusType.OK);
                        return;
                    }
                } catch (Exception e) {
                    log.warn("Minio cache is not available", e);
                }
                log.debug("Cached artifact for " + stage + " unavailable: " + uri);
            }
        }

        String bucket = stageParams.getOrDefault("bucket", "workflow");
        String dir = String.join("/", "workflow-" + workflow.getId(), UUID.randomUUID().toString());
        String prefix = "s3://" + bucket + "/" + dir + "/" + "area-";

        createRasterData(aoi, prefix + aoi.getId() + ".tif", cacheKey);

        super.start(stage);
    }

    private void useLocalData(Stage stage, RasterSource rasterSource) {
        Workflow workflow = stage.getWorkflow();

        Map<String, String> params = rasterSource.getParams();
        if (params.containsKey("url")) {
            String url = params.get("url");

            AreaOfInterest aoi = areaOfInterestRepository.findByWorkflow(workflow)
                    .orElseThrow(() -> new RuntimeException("No area of interest found for " + workflow));

            createRasterData(aoi, url, null);

            stage.getStageStatus().setStatus(StatusType.OK);
        } else {
            throw new RuntimeException("Param 'url' is not specified for " + stage);
        }
    }

    private void createRasterData(AreaOfInterest aoi, String uri, String cacheKey) {
        Artifact artifact = new Artifact(aoi.getWorkflow(), aoi, ArtifactType.RAW_RASTER, uri);
        artifact.setCacheKey(cacheKey);

        artifactRepository.save(artifact);
    }
}
