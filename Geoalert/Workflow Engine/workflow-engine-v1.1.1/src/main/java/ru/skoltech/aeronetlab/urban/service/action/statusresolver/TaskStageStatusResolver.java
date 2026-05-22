package ru.skoltech.aeronetlab.urban.service.action.statusresolver;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.workflow.Message;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.TaskStatus;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskRepository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class TaskStageStatusResolver implements StageStatusResolver {

    @Autowired
    private TaskRepository taskRepository;

    @Override
    public void resolve(StageStatus stageStatus) {
        List<TaskStatus> taskStatuses = taskRepository.findAllByStage(stageStatus.getStage())
                .stream()
                .map(task ->
                        Optional.ofNullable(task.getTaskStatus()).orElseThrow(() ->
                                new IllegalStateException("" + task + " has no status.")
                        )
                )
                .collect(Collectors.toList());

        List<StatusType> statuses = taskStatuses.stream().map(TaskStatus::getStatus).collect(Collectors.toList());

        if (statuses.contains(StatusType.FAILED)) {
            String msg = taskStatuses.stream()
                    .filter(s -> s.getStatus() == s.getStatus())
                    .map(TaskStatus::getMessage)
                    .filter(m -> m != null && !m.isEmpty())
                    .collect(Collectors.joining("\n"));

            List<Message> messages = taskStatuses.stream()
                    .filter(s -> s.getStatus() == s.getStatus())
                    .flatMap(s -> s.getMessages().stream())
                    .collect(Collectors.toList());
            stageStatus.setMessages(messages);
            stageStatus.setErrorMessage(msg.isEmpty() ? null : msg);
            stageStatus.setStatus(StatusType.FAILED);
        } else if (statuses.contains(StatusType.WAITING_TO_RETRY)) {
            stageStatus.setStatus(StatusType.IN_PROGRESS);
        } else if (statuses.contains(StatusType.IN_PROGRESS)) {
            stageStatus.setStatus(StatusType.IN_PROGRESS);
        } else if (statuses.stream().allMatch(s -> s == StatusType.OK)) {
            stageStatus.setStatus(StatusType.OK);
        } else {
            throw new IllegalStateException("Couldn't determine status of " + stageStatus.getStage());
        }
    }
}
