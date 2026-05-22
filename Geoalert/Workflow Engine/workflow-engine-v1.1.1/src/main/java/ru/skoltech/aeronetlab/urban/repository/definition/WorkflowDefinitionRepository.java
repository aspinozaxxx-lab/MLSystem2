package ru.skoltech.aeronetlab.urban.repository.definition;


import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;

import java.util.Optional;

public interface WorkflowDefinitionRepository extends CrudRepository<WorkflowDefinition, Long> {

    Optional<WorkflowDefinition> findByName(String name);
}
