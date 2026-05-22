package ru.skoltech.aeronetlab.urban.service.workflow;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.workflow.StageRepository;
import ru.skoltech.aeronetlab.urban.service.definition.StageDefinitionService;
import ru.skoltech.aeronetlab.urban.service.definition.TopologicalSorter;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class StageService {

    @Autowired
    private StageRepository stageRepository;

    @Autowired
    private StageDefinitionService stageDefinitionService;

    @Autowired
    private StageStatusService stageStatusService;

    @Autowired
    private TaskService taskService;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    public List<Stage> getTopologicalSorting(Workflow workflow) {
        Set<Stage> stages = stageRepository.findAllByWorkflow(workflow);
        return new TopologicalSorter<>(stages, Stage::getPreviousStages).sort();
    }

    public List<Stage> getTopologicalSortingStartingWith(Workflow workflow,
                                                         Set<Stage> roots) {
        Set<Stage> stages = stageRepository.findAllByWorkflow(workflow);
        return new TopologicalSorter<>(stages, Stage::getPreviousStages)
                .sortStartingWith(roots);
    }

    public void createStages(Workflow workflow) {
        Map<StageDefinition, Stage> currentStages = new HashMap<>();
        for (Stage existing : stageRepository.findAllByWorkflow(workflow)) {
            currentStages.put(existing.getStageDefinition(), existing);
        }

        WorkflowDefinitionVer workflowDefVer = workflow.getWorkflowDefinitionVer();
        List<StageDefinition> sorting = stageDefinitionService.getTopologicalSorting(workflowDefVer);

        for (StageDefinition stageDef : sorting) {
            // if partially recreating stages, skip already existing
            if (currentStages.containsKey(stageDef)) {
                continue;
            }

            Set<Stage> prevStages = stageDef.getPreviousStages()
                    .stream()
                    .map(currentStages::get)
                    .collect(Collectors.toSet());
            currentStages.put(stageDef, create(stageDef, prevStages, workflow));
        }
    }

    private Stage create(StageDefinition stageDefinition,
                         Set<Stage> previousStages,
                         Workflow workflow) {
        Stage stage = new Stage(previousStages);
        stage.setWorkflow(workflow);
        stage.setStageDefinition(stageDefinition);

        stage = stageRepository.save(stage);

        log.info("Created new " + stage + " with " + previousStages.size() + " parents");

        stageStatusService.createPending(stage);

        return stage;
    }

    public void deleteStartingWith(Workflow workflow, Set<Stage> stages) {
        List<Stage> reverseSorting = getTopologicalSortingStartingWith(workflow, stages);
        Collections.reverse(reverseSorting);

        for (Stage stage : reverseSorting) {
            log.info("Deleting " + stage);

            taskService.deleteTasks(stage);
            stageStatusService.delete(stage);
            stageRepository.delete(stage);
        }
    }
}
