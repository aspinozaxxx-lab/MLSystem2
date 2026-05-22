package ru.skoltech.aeronetlab.urban.repository.workflow;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.util.Optional;

public interface AreaOfInterestRepository extends CrudRepository<AreaOfInterest, Long> {
    Optional<AreaOfInterest> findByWorkflow(Workflow workflow);
}
