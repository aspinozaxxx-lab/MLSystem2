package ru.skoltech.aeronetlab.urban.repository.definition;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;

import java.util.Set;

public interface StageDefinitionRepository extends CrudRepository<StageDefinition, Long> {

   default Set<StageDefinition> findNextStages(StageDefinition previousStage) {
       return findAllByPreviousStages(previousStage);
   }

    default Set<StageDefinition> findFirstStages(WorkflowDefinitionVer workflowDefinitionVer) {
        return findAllByWorkflowDefinitionVerAndPreviousStagesEmpty(workflowDefinitionVer);
    }

   Set<StageDefinition> findAllByWorkflowDefinitionVer(WorkflowDefinitionVer workflowDefinitionVer);

   Set<StageDefinition> findAllByPreviousStages(StageDefinition previousStage);

    Set<StageDefinition> findAllByWorkflowDefinitionVerAndPreviousStagesEmpty(WorkflowDefinitionVer workflowDefinitionVer);
}
