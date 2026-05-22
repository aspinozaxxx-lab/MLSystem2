package ru.skoltech.aeronetlab.urban.service.watchdog;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.workflow.Message;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.workflow.*;
import ru.skoltech.aeronetlab.urban.service.workflow.WorkflowStatusService;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class WatchdogService {
    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Value("#{systemEnvironment['WORKFLOW_TIMEOUT_HOURS'] ?: '24'}")
    private int workflowTimeout = 24;

    @Value("#{systemEnvironment['USER_INPUT_TIMEOUT_HOURS'] ?: '72'}")
    private  int userInputTimeoutHours = 72;

    @Autowired
    private WorkflowRepository workflowRepository;

    @Autowired
    private StageRepository stageRepository;

    @Autowired
    private WorkflowStatusService workflowStatusService;

    @Transactional(propagation = Propagation.REQUIRED)
    public void terminateStuckWorkflows(LocalDateTime now) {
        Collection<Workflow> workflows = workflowRepository.findStuckByStatus(
                StatusType.IN_PROGRESS, now.minus(workflowTimeout, ChronoUnit.HOURS));

        for (Workflow workflow : workflows) {
            try {
                terminateWorkflow(workflow, now);
                log.info("Workflow " + workflow.getId() + " was terminated after " + workflowTimeout + " hours");
            } catch (Exception ex) {
                log.error("Unable to terminate workflow " + workflow.getId(), ex);
            }

        }
    }

    @Transactional(propagation = Propagation.SUPPORTS)
    public void terminateWorkflow(Workflow workflow, LocalDateTime now) {
        long interval = ChronoUnit.HOURS.between(workflow.getWorkflowStatus().getUpdateDate(), now);

        Set<Stage> stages = stageRepository.findAllByWorkflow(workflow);

        Optional<Stage> userInputStageOption = stages.stream()
                .filter(stage -> stage.getStageDefinition().getAction().equals(Action.USER_INPUT))
                .findAny();

        if (userInputStageOption.isPresent()) {
            Stage userInput = userInputStageOption.get();
            Optional<LocalDateTime> stageStart = Optional.ofNullable(userInput.getStageStatus().getStartDate());
            long inputInterval = ChronoUnit.HOURS.between(stageStart.orElse(workflow.getCreateDate()), now);
            if (inputInterval < userInputTimeoutHours + workflowTimeout) {
                //Increase timeout if workflow has user-input stage
                return;
            }
        }

        String action = stages.stream()
                .filter(s -> s.getStageStatus() != null)
                .filter(stage -> StatusType.IN_PROGRESS.equals(stage.getStageStatus().getStatus()))
                .map(s -> s.getStageDefinition().getAction().getActionName()).collect(Collectors.joining(", "));

        String msg = "Workflow " + workflow.getId() + " is stuck IN_PROCESSING for " + interval + " hours at stage " + action;
        log.info(msg);

        Map<String, String> params = new HashMap<>();
        params.put("interval", String.valueOf(interval));
        params.put("action", action);

        workflowStatusService.terminateFailed(now, workflow, new Message("workflow_engine.workflowTimeout", params, msg));
    }
}
