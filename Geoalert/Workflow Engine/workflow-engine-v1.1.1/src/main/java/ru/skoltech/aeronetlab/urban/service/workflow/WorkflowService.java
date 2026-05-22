package ru.skoltech.aeronetlab.urban.service.workflow;

import com.google.common.collect.Sets;
import org.apache.commons.collections.CollectionUtils;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.api.exception.BadRequestException;
import ru.skoltech.aeronetlab.urban.api.exception.NotFoundException;
import ru.skoltech.aeronetlab.urban.dto.business.AreaOfInterestDto;
import ru.skoltech.aeronetlab.urban.dto.workflow.WorkflowDto;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.ArtifactRepository;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionRepository;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionVerRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.AreaOfInterestRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.StageStatusRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskStatusRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowStatusRepository;
import ru.skoltech.aeronetlab.urban.service.queue.status.StatusUpdateSender;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class WorkflowService {

    @Autowired
    private AreaOfInterestRepository areaOfInterestRepository;

    @Autowired
    private ArtifactRepository artifactRepository;

    @Autowired
    private WorkflowRepository workflowRepository;

    @Autowired
    private StageService stageService;

    @Autowired
    private StageStatusRepository stageStatusRepository;

    @Autowired
    private TaskStatusRepository taskStatusRepository;

    @Autowired
    private WorkflowStatusService workflowStatusService;

    @Autowired
    private WorkflowStatusRepository workflowStatusRepository;

    @Autowired
    private WorkflowDefinitionVerRepository workflowDefinitionVerRepository;

    @Autowired
    private WorkflowDefinitionRepository workflowDefinitionRepository;

    @Autowired
    private StatusUpdateSender statusUpdateSender;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Value("${max.aoi.area.degrees:10}") // ~100000 sq km
    private Double maxAoiArea;

    @Transactional
    public Workflow create(WorkflowDto dto) {
        List<Geometry> geometries = dto.getAreasOfInterest()
                .stream()
                .map(AreaOfInterestDto::getGeometry)
                .toList();

        geometries.forEach(this::validateGeometry);

        WorkflowDefinition wd = workflowDefinitionRepository.findById(dto.getWorkflowDefinitionId())
                .orElseThrow(() -> new NotFoundException(WorkflowDefinition.class, dto.getWorkflowDefinitionId()));
        WorkflowDefinitionVer wdVer = workflowDefinitionVerRepository.findLatest(wd)
                .orElseThrow(() -> new RuntimeException("No versions exist for " + wd));

        Workflow workflow = new Workflow(wdVer);

        dto.getParams().forEach((k, v) -> workflow.getParams().put(k, v));
        dto.getMeta().forEach((k, v) -> workflow.getMeta().put(k, v));

        if (dto.getBlocks() != null) {
            workflow.setBlockParams(dto.getBlocks());
        }

        if (dto.getSystem() == null) {
            throw new BadRequestException("'system' parameter is required");
        }
        workflow.setSystem(dto.getSystem());
        workflow.setProcessingId(dto.getProcessingId());

        Set<AreaOfInterest> aois = geometries.stream()
                .map(g -> new AreaOfInterest(workflow, g))
                .collect(Collectors.toSet());
        areaOfInterestRepository.saveAll(aois);

        if (CollectionUtils.isNotEmpty(dto.getArtifacts())) {
            AreaOfInterest firstAoi = aois.stream().findFirst().orElse(null);

            Set<Artifact> artifacts = dto.getArtifacts().stream()
                    .map(artifactDto -> new Artifact(workflow, firstAoi, ArtifactType.valueOf(artifactDto.getArtifactType()), artifactDto.getUri()))
                    .collect(Collectors.toSet());
            artifactRepository.saveAll(artifacts);
        }



        Workflow workflowFinal = workflowRepository.save(workflow);

        log.info("Created new " + workflowFinal);

        stageService.createStages(workflow);

        workflowStatusService.setInProgress(workflowFinal);

        statusUpdateSender.send(workflowFinal);

        return workflowFinal;
    }

    private void validateGeometry(Geometry geometry) {
        Envelope bbox = geometry.getEnvelopeInternal();
            if (bbox.getMinX() < -360 * 3 ||
                    bbox.getMinX() > 360 * 3 ||
                    bbox.getMaxX() < -360 * 3 ||
                    bbox.getMaxX() > 360 * 3) {
                throw new BadRequestException("The geometry is most likely in wrong crs.");
            }

        if (bbox.getMinY() < -360 * 3 ||
                bbox.getMinY() > 360 * 3 ||
                bbox.getMaxY() < -360 * 3 ||
                bbox.getMaxY() > 360 * 3) {
            throw new BadRequestException("The geometry is most likely in wrong crs.");
        }

        if (geometry.getArea() > maxAoiArea) {
            throw new BadRequestException("Too large geometry.");
        }
    }

    @Transactional
    public void restart(Workflow workflow) {
        Set<StatusType> statuses = new HashSet<>(Arrays.asList(StatusType.values()));
        restartWithStatuses(workflow, statuses);
    }

    @Transactional
    public void restartFromFailed(Workflow workflow) {
        Set<StatusType> failed = Sets.newHashSet(StatusType.FAILED, StatusType.CANCELLED);
        restartWithStatuses(workflow, failed);
    }

    @Transactional
    public void restartStartingWith(Set<Stage> stages) {
        log.info("Deleting and recreating stages starting with " + stages);

        if (stages.isEmpty()) {
            return;
        }

        Workflow workflow = stages.iterator().next().getWorkflow();

        stageService.deleteStartingWith(workflow, stages);

        stageService.createStages(workflow);

        workflowStatusService.setInProgress(workflow);

        statusUpdateSender.send(workflow);
    }

    @Transactional
    public void restartWithStatuses(Workflow workflow, Set<StatusType> statuses) {
        Set<Stage> stages = stageStatusRepository.findAllByWorkflowAndStatuses(workflow, statuses)
                .stream()
                .map(StageStatus::getStage)
                .collect(Collectors.toSet());
        restartStartingWith(stages);
    }

    @Transactional
    public void cancel(Set<Long> ids) {
        LocalDateTime now = LocalDateTime.now();

        workflowStatusRepository.cancelWorkflows(ids, now);
        taskStatusRepository.cancelStages(ids, now);
        stageStatusRepository.cancelStages(ids, now);
    }
}
