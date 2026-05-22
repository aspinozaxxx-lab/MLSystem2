package ru.skoltech.aeronetlab.urban.repository.workflow;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.util.Set;

public interface StageRepository extends CrudRepository<Stage, Long> {
    Set<Stage> findAllByWorkflow(Workflow workflow);
}
