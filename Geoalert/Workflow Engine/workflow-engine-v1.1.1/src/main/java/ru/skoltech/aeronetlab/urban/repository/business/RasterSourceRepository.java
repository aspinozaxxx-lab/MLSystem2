package ru.skoltech.aeronetlab.urban.repository.business;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSource;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.util.Optional;

public interface RasterSourceRepository extends CrudRepository<RasterSource, Long> {

    Optional<RasterSource> findByWorkflow(Workflow workflow);
}
