package ru.skoltech.aeronetlab.urban.service.workflow;

import com.google.common.collect.Sets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.entity.workflow.*;
import ru.skoltech.aeronetlab.urban.repository.workflow.StageStatusRepository;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.StageStarters;
import ru.skoltech.aeronetlab.urban.service.action.statusresolver.StageStatusResolver;
import ru.skoltech.aeronetlab.urban.service.action.statusresolver.StageStatusResolvers;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.Set;

@Service
public class StageStatusService {

    @Autowired
    private StageStatusRepository stageStatusRepository;

    @Autowired
    private StageStatusResolvers stageStatusResolvers;

    @Autowired
    private StageStarters stageStarters;

    @Autowired
    private TaskStatusService taskStatusService;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Transactional(propagation = Propagation.SUPPORTS)
    public StageStatus updateStatus(Stage stage) {
        StageStatus statusEntity = Optional.ofNullable(stage.getStageStatus())
                .orElseThrow(() -> new RuntimeException("No status exists for " + stage));
        StatusType currStatus = statusEntity.getStatus();

        if (currStatus == StatusType.IN_PROGRESS) {
            try {
                StageStatusResolver statusResolver = stageStatusResolvers.get(stage.getStageDefinition().getAction());
                statusResolver.resolve(statusEntity);
            } catch (Throwable e) {
                String msg = "Error resolving status of " + stage + ":" + e;
                log.error(msg, e);
                statusEntity.setStatus(StatusType.FAILED);
                statusEntity.setErrorMessage(msg);
                statusEntity.getMessages().add(Message.internalError(msg));
            }
        } else if (currStatus == StatusType.PENDING) {
            boolean canStart = stage.getPreviousStages()
                    .stream()
                    .map(sv -> Optional.ofNullable(sv.getStageStatus())
                            .map(StageStatus::getStatus)
                            .orElseThrow(() -> new RuntimeException("No status exists for " + stage)))
                    .allMatch(s -> s == StatusType.OK);

            if (canStart) {
                statusEntity.setStatus(StatusType.IN_PROGRESS);
                statusEntity.setStartDate(LocalDateTime.now());
                log.info("Starting stage: " + stage);
                try {
                    stageStarters.get(stage.getStageDefinition().getAction()).start(stage);
                } catch (Throwable e) {
                    String msg = "Error trying to start " + stage + ": " + e;
                    log.error(msg, e);
                    statusEntity.setStatus(StatusType.FAILED);
                    statusEntity.setErrorMessage(msg);
                    statusEntity.getMessages().add(Message.internalError(msg));
                }
            }
        }

        if (statusEntity.getStatus() != currStatus) {
            statusEntity.setUpdateDate(LocalDateTime.now());
            statusEntity = stageStatusRepository.save(statusEntity);
            log.info("Updated status: " + statusEntity);
        }

        return statusEntity;
    }

    public void createPending(Stage stage) {
        StageStatus stageStatus = new StageStatus(stage);
        stageStatus.setStatus(StatusType.PENDING);
        stageStatus.setUpdateDate(LocalDateTime.now());

        stageStatus = stageStatusRepository.save(stageStatus);
        stage.setStageStatus(stageStatus);

        log.info("Created new " + stageStatus);
    }

    public void delete(Stage stage) {
        Optional.ofNullable(stage.getStageStatus()).ifPresent(s -> {
            log.info("Deleting " + s);
            stageStatusRepository.delete(s);
        });
    }

    @Transactional(propagation = Propagation.REQUIRED)
    public void terminateFailed(LocalDateTime now, Workflow workflow, Message message) {
        Set<StageStatus> statuses = stageStatusRepository.findAllByWorkflowAndStatuses(workflow,
                Sets.newHashSet(StatusType.WAITING_TO_RETRY, StatusType.IN_PROGRESS));

        for (StageStatus status : statuses) {
            status.setStatus(StatusType.FAILED);
            status.setUpdateDate(now);
            status.getMessages().add(message);

            taskStatusService.terminateFailed(now, status.getStage(), message);
        }

        stageStatusRepository.saveAll(statuses);
    }

}
