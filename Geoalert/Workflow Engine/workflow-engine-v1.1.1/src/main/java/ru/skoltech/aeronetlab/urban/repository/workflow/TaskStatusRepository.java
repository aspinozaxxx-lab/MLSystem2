package ru.skoltech.aeronetlab.urban.repository.workflow;

import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.TaskStatus;

import java.time.LocalDateTime;
import java.util.Set;

public interface TaskStatusRepository extends CrudRepository<TaskStatus, Long> {

    Set<TaskStatus> findByStatus(StatusType status);

    @Modifying
    @Query("update TaskStatus ts " +
            "set ts.status = ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.CANCELLED, " +
            "ts.updateDate = ?2 " +
            "where ts.status in (ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.IN_PROGRESS, " +
            "ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.WAITING_TO_RETRY) " +
            "and ts.task in (select tk from Task tk where tk.stage.workflow.id in (?1))")
    void cancelStages(Set<Long> ids, LocalDateTime updateDate);
}
