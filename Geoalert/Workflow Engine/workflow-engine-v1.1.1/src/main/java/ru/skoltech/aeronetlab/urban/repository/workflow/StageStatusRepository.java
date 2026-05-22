package ru.skoltech.aeronetlab.urban.repository.workflow;

import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public interface StageStatusRepository extends CrudRepository<StageStatus, Long> {

    default Set<StageStatus> findWaitingForUserInput() {
        List<Action> userInputActions = Arrays.stream(Action.values())
                .filter(Action::requiresUserInput)
                .collect(Collectors.toList());
        return findByActionsAndStatus(userInputActions, StatusType.IN_PROGRESS);
    }

    @Query("select ss from StageStatus as ss " +
            "join ss.stage as st " +
            "join st.stageDefinition as sd " +
            "where sd.action in (?1) and ss.status = ?2")
    Set<StageStatus> findByActionsAndStatus(List<Action> actions, StatusType status);

    @Query("select ss from StageStatus as ss " +
            "join ss.stage as st " +
            "where st.workflow = ?1 and ss.status in (?2)")
    Set<StageStatus> findAllByWorkflowAndStatuses(Workflow workflow, Set<StatusType> statuses);

    @Modifying
    @Query("update StageStatus ss " +
            "set ss.status = ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.CANCELLED, " +
            "ss.updateDate = ?2 " +
            "where ss.status in (ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.IN_PROGRESS, " +
            "ru.skoltech.aeronetlab.urban.entity.workflow.StatusType.PENDING) " +
            "and ss.stage in (select st from Stage st where st.workflow.id in (?1))")
    void cancelStages(Set<Long> ids, LocalDateTime updateDate);
}
