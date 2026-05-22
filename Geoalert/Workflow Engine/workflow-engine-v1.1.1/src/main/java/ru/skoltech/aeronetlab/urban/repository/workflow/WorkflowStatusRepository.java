package ru.skoltech.aeronetlab.urban.repository.workflow;

import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.workflow.WorkflowStatus;

import java.time.LocalDateTime;
import java.util.Set;

public interface WorkflowStatusRepository extends CrudRepository<WorkflowStatus, Long> {

    @Modifying
    @Query("update WorkflowStatus ws " +
            "set ws.status = ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.CANCELLED, " +
            "ws.updateDate = ?2 " +
            "where ws.status = ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.IN_PROGRESS " +
            "and ws.workflow.id in (?1)")
    void cancelWorkflows(Set<Long> ids, LocalDateTime updateDate);
}
