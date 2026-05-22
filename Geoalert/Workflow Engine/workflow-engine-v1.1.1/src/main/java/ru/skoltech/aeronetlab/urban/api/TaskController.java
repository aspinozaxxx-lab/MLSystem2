package ru.skoltech.aeronetlab.urban.api;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.skoltech.aeronetlab.urban.dto.workflow.TaskDto;
import ru.skoltech.aeronetlab.urban.service.mapper.workflow.TaskMapper;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;

@RestController
@RequestMapping(path = "api/v0/tasks")
public class TaskController {

  @Autowired
  private TaskMapper taskMapper;

  @GetMapping("/{taskId}")
  public ResponseEntity<TaskDto> getTask(@PathVariable Long taskId) {
    return ResponseEntity.ok(taskMapper.getTask(taskId));
  }

  @GetMapping("/{taskId}/runcheck")
  public ResponseEntity<Integer> runCheck(@PathVariable Long taskId) {
    TaskDto task = taskMapper.getTask(taskId);

    if (StatusType.valueOf(task.getStatus()) == StatusType.IN_PROGRESS) {
      return ResponseEntity.ok(0);
    }

    return ResponseEntity.ok(1);
  }
}
