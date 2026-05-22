package ru.skoltech.aeronetlab.urban.repository.definition;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;

import java.util.Optional;
import java.util.Set;

public interface WorkflowDefinitionVerRepository extends CrudRepository<WorkflowDefinitionVer, Long> {

    default Optional<WorkflowDefinitionVer> findLatest(WorkflowDefinition workflowDefinition) {
        return findTop1ByWorkflowDefinitionOrderByVersionDesc(workflowDefinition);
    }

    Set<WorkflowDefinitionVer> findAllByWorkflowDefinition(WorkflowDefinition workflowDefinition);

    Optional<WorkflowDefinitionVer> findTop1ByWorkflowDefinitionOrderByVersionDesc(WorkflowDefinition workflowDefinition);
}
