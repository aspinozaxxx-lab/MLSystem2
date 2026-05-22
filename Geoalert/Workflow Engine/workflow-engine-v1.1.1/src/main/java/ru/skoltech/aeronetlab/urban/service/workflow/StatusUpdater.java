package ru.skoltech.aeronetlab.urban.service.workflow;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.api.exception.NotFoundException;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.workflow.StageStatusRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskStatusRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowRepository;
import ru.skoltech.aeronetlab.urban.service.queue.status.StatusUpdateSender;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Collections;
import java.util.Optional;

@Service
public class StatusUpdater {

    @Autowired
    private TaskStatusService taskStatusService;

    @Autowired
    private TaskStatusRepository taskStatusRepository;

    @Autowired
    private TaskRepository taskRepository;

    @Autowired
    private WorkflowStatusService workflowStatusService;

    @Autowired
    private WorkflowRepository workflowRepository;

    @Autowired
    private StageStatusRepository stageStatusRepository;

    @Autowired
    private StageStatusService stageStatusService;

    @Autowired
    private StatusUpdateSender statusUpdateSender;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Scheduled(fixedDelay = 20_000)
    @Transactional
    public void updateStatuses() {
        stageStatusRepository.findWaitingForUserInput()
                .forEach(ss -> updateStage(ss.getStage()));

        taskStatusRepository.findByStatus(StatusType.WAITING_TO_RETRY)
                .forEach(s -> updateWaitingToRetry(s.getTask().getId()));
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    protected void updateStage(Stage stage) {
        StageStatus status = stageStatusService.updateStatus(stage);
        if (status.getStatus() != StatusType.IN_PROGRESS) {
            statusUpdateSender.send(stage.getWorkflow());
        }
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    protected void updateWaitingToRetry(Long taskId) {
        taskRepository.findById(taskId).ifPresent(task -> {
            int interval = Optional.of(task.getStage().getStageDefinition().getRetryInterval())
                    .orElse(60);
            if (task.getTaskStatus().getUpdateDate().until(LocalDateTime.now(), ChronoUnit.SECONDS) >= interval) {
                taskStatusService.updateStatus(task, StatusType.IN_PROGRESS, Optional.empty(), Collections.emptyList(), Collections.emptyMap());
            }
        });
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void updateWorkflow(long workflowId) {
        Workflow workflow = workflowRepository.findById(workflowId)
                .orElseThrow(() -> new NotFoundException(Workflow.class, workflowId));

        workflowStatusService.updateStatus(workflow);
    }
}
