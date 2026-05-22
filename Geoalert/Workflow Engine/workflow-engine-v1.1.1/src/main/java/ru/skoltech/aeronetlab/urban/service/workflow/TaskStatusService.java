package ru.skoltech.aeronetlab.urban.service.workflow;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.entity.workflow.*;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskStatusRepository;
import ru.skoltech.aeronetlab.urban.service.queue.MessageSender;
import ru.skoltech.aeronetlab.urban.service.queue.ResponseMessage;
import ru.skoltech.aeronetlab.urban.service.queue.status.StatusUpdateSender;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class TaskStatusService {

    @Autowired
    private TaskRepository taskRepository;

    @Autowired
    private TaskStatusRepository taskStatusRepository;

    @Autowired
    private MessageSender messageSender;

    @Autowired
    private StatusUpdateSender statusUpdateSender;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    public void updateStatus(Task task, StatusType status, Optional<String> errorMessage, List<Message> messages, Map<String, String> results) {
        TaskStatus currStatusEntity = task.getTaskStatus() == null ? new TaskStatus(task) : task.getTaskStatus();

        if (status.equals(currStatusEntity.getStatus())) return;
        if (currStatusEntity.getStatus() == StatusType.CANCELLED) return;

        if (status == StatusType.OK) {
            currStatusEntity.setMessage(null);
            currStatusEntity.setStatus(status);
            currStatusEntity.getMessages().addAll(messages);
        } else if (status == StatusType.FAILED) {
            errorMessage.ifPresent(currStatusEntity::setMessage);

            int maxRetries = Optional.ofNullable(task.getStage().getStageDefinition().getRetries()).orElse(0);
            int currAttempts = Optional.ofNullable(currStatusEntity.getAttempts()).orElse(1);
            if (currAttempts - 1 < maxRetries) {
                currStatusEntity.setStatus(StatusType.WAITING_TO_RETRY);
            } else {
                currStatusEntity.setStatus(StatusType.FAILED);
            }
            currStatusEntity.getMessages().addAll(messages);
        } else if (status == StatusType.IN_PROGRESS) {
            currStatusEntity.setStatus(status);
            if (currStatusEntity.getStartDate() == null) {
                currStatusEntity.setStartDate(LocalDateTime.now());
            }
            currStatusEntity.setAttempts(Optional.ofNullable(currStatusEntity.getAttempts()).orElse(0) + 1);
            messageSender.send(task);
        } else {
            currStatusEntity.setStatus(status);
        }

        currStatusEntity.setUpdateDate(LocalDateTime.now());
        currStatusEntity.setResults(results);

        TaskStatus finalStatusEntity = taskStatusRepository.save(currStatusEntity);
        task.setTaskStatus(finalStatusEntity);

        log.info("Updated status: " + finalStatusEntity);
    }

    @Transactional(propagation = Propagation.SUPPORTS)
    public void updateStatus(ResponseMessage responseMessage) {
        Task task = taskRepository.findById(responseMessage.getTaskId())
                .orElseThrow(() -> new RuntimeException("Couldn't find task with id=" + responseMessage.getTaskId()));

        StatusType status = responseMessage.getStatus() == 0 ? StatusType.OK : StatusType.FAILED;

        List<Message> messages = Optional.ofNullable(responseMessage.getMessages())
                .orElse(Collections.emptyList())
                .stream()
                .map(message -> new Message(message.getCode(), message.getParameters(), StringUtils.truncate(message.getMessage(), 1024)))
                        .collect(Collectors.toList());

        Optional<String> errorMessage = Optional.ofNullable(responseMessage.getErrorMessage());
        errorMessage.ifPresent((msg) -> {
            String worker = task.getStage().getStageDefinition().getAction().getWorkerName().orElse("unknownWorker");
            messages.add(new Message(worker + ".internalError", Collections.emptyMap(), msg));
        });

        updateStatus(task, status, errorMessage, messages, responseMessage.getResults());

        statusUpdateSender.send(task.getStage().getWorkflow());
    }

    @Transactional(propagation = Propagation.REQUIRED)
    public void terminateFailed(LocalDateTime now, Stage stage, Message message) {
        Set<Task> tasks = taskRepository.findAllByStage(stage);
        for (Task task : tasks) {
            TaskStatus status = task.getTaskStatus();
            status.setStatus(StatusType.FAILED);
            status.setUpdateDate(now);
            status.getMessages().add(message);
        }

        taskRepository.saveAll(tasks);

    }
}
