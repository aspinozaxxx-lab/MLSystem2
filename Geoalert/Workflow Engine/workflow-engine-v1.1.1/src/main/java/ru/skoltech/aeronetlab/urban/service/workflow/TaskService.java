package ru.skoltech.aeronetlab.urban.service.workflow;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskStatusRepository;
import ru.skoltech.aeronetlab.urban.service.queue.TaskMessage;

import java.util.Collections;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

@Service
public class TaskService {

  @Autowired
  private TaskRepository taskRepository;

  @Autowired
  private TaskStatusRepository taskStatusRepository;

  @Autowired
  private ObjectMapper objectMapper;

  @Autowired
  private TaskStatusService taskStatusService;

  private final Logger log = LoggerFactory.getLogger(this.getClass());

  @Value("${runcheck.url:http://engine.workflow-duty.svc:8080}")
  private String runcheckUrl;

  public Task create(Stage stage, AreaOfInterest aoi, TaskMessage taskMessage) {

    Task task = new Task(stage, aoi);
    task = taskRepository.save(task);

    taskMessage.setTask_id(task.getId());
    taskMessage.setProcessing_id(stage.getWorkflow().getProcessingId());
    taskMessage.setRuncheck_url(task.getId(), runcheckUrl);

    try {
      task.setRequest(objectMapper.writeValueAsString(taskMessage));
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Error serializing task message: " + e, e);
    }

    taskRepository.save(task);

    log.info("Created new " + task);

    taskStatusService.updateStatus(task, StatusType.IN_PROGRESS, Optional.empty(), Collections.emptyList(), Collections.emptyMap());

    return task;
  }

  public void deleteTasks(Stage stage) {
    Set<Task> tasks = taskRepository.findAllByStage(stage);

    tasks.stream()
        .map(Task::getTaskStatus)
        .filter(Objects::nonNull)
        .peek(ts -> log.info("Deleting " + ts))
        .forEach(ts -> taskStatusRepository.delete(ts));

    tasks.stream()
        .peek(t -> log.info("Deleting " + t))
        .forEach(taskRepository::delete);
  }
}
