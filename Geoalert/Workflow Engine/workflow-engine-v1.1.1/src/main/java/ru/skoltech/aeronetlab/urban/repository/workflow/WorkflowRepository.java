package ru.skoltech.aeronetlab.urban.repository.workflow;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.time.LocalDateTime;
import java.util.Collection;

public interface WorkflowRepository extends CrudRepository<Workflow, Long> {

    @Query("select w from Workflow as w " +
            "join w.workflowStatus as ws " +
            "where ws.status = ?1 and ws.updateDate < ?2")
    Collection<Workflow> findStuckByStatus(StatusType statusType, LocalDateTime updatedBefore);

    @Query("select w from Workflow as w " +
            "where w.processingId = ?1")
    Collection<Workflow> findByProcessingId(String processingId);
}
