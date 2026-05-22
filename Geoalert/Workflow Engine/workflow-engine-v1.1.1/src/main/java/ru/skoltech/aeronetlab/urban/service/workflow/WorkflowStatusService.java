package ru.skoltech.aeronetlab.urban.service.workflow;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.entity.workflow.*;
import ru.skoltech.aeronetlab.urban.repository.workflow.StageRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowStatusRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class WorkflowStatusService {

    @Autowired
    private StageRepository stageRepository;

    @Autowired
    private WorkflowStatusRepository workflowStatusRepository;

    @Autowired
    private StageService stageService;

    @Autowired
    private StageStatusService stageStatusService;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    public void setInProgress(Workflow workflow) {
        Optional<WorkflowStatus> workflowStatusOptional = Optional.ofNullable(workflow.getWorkflowStatus());
        WorkflowStatus workflowStatus = workflowStatusOptional.orElseGet(() -> new WorkflowStatus(workflow));
        workflowStatus.setStatus(StatusType.IN_PROGRESS);
        workflowStatus.setUpdateDate(LocalDateTime.now());

        workflowStatus = workflowStatusRepository.save(workflowStatus);
        workflow.setWorkflowStatus(workflowStatus);

        if (workflowStatusOptional.isPresent()) {
            log.info("Updated " + workflowStatus);
        } else {
            log.info("Created new " + workflowStatus);
        }
    }

    @Transactional(propagation = Propagation.SUPPORTS)
    public void updateStatus(Workflow workflow) {
        WorkflowStatus statusEntity = Optional.ofNullable(workflow.getWorkflowStatus())
                .orElseThrow(() -> new RuntimeException("No status exists for " + workflow));
        StatusType currStatus = statusEntity.getStatus();

        if (currStatus == StatusType.CANCELLED) {
            return;
        }

        updateStageStatuses(workflow);

        List<StatusType> stageStatuses = stageRepository.findAllByWorkflow(workflow)
                .stream()
                .map(Stage::getStageStatus)
                .filter(Objects::nonNull)
                .map(StageStatus::getStatus)
                .collect(Collectors.toList());

        StatusType newStatus = currStatus;
        if (stageStatuses.contains(StatusType.FAILED)) {
            newStatus = StatusType.FAILED;
        } else if (stageStatuses.stream().allMatch(s -> s == StatusType.OK)) {
            newStatus = StatusType.OK;
        }

        if (newStatus != currStatus) {
            statusEntity.setUpdateDate(LocalDateTime.now());
            statusEntity.setStatus(newStatus);
            statusEntity = workflowStatusRepository.save(statusEntity);
            log.info("Updated status: " + statusEntity);
        }
    }

    private void updateStageStatuses(Workflow workflow) {
        List<Stage> sorting = stageService.getTopologicalSorting(workflow);
        for (Stage stage : sorting) {
            stageStatusService.updateStatus(stage);
        }
    }

    @Transactional(propagation = Propagation.SUPPORTS)
    public void terminateFailed(LocalDateTime now, Workflow workflow, Message message) {
        Optional<WorkflowStatus> workflowStatusOptional = Optional.ofNullable(workflow.getWorkflowStatus());
        WorkflowStatus workflowStatus = workflowStatusOptional.orElseGet(() -> new WorkflowStatus(workflow));
        workflowStatus.setStatus(StatusType.FAILED);
        workflowStatus.setUpdateDate(now);

        workflowStatus = workflowStatusRepository.save(workflowStatus);
        workflow.setWorkflowStatus(workflowStatus);

        if (workflowStatusOptional.isPresent()) {
            log.info("Updated " + workflowStatus);
        } else {
            log.info("Created new " + workflowStatus);
        }

        stageStatusService.terminateFailed(now, workflow, message);
    }
}
