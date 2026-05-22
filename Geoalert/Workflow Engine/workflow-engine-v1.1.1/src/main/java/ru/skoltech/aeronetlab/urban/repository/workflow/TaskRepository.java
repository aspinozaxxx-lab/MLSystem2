package ru.skoltech.aeronetlab.urban.repository.workflow;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;

import java.util.Set;

public interface TaskRepository extends CrudRepository<Task, Long> {

    Set<Task> findAllByStage(Stage stage);
}
