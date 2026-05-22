package ru.skoltech.aeronetlab.urban.service.action.taskcreator;

import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.definition.action.WithTaskCreator;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;

import java.lang.reflect.Field;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

@Service
public class TaskCreators {
    private final Map<Action, TaskCreator> taskCreatorsMap = new HashMap<>();

    public TaskCreators(Collection<TaskCreator> taskCreators) {
        Map<Action, Class<? extends TaskCreator>> classes = new HashMap<>();
        for (Action action : Action.values()) {
            try {
                Field field = Action.class.getField(action.name());
                if (field.isAnnotationPresent(WithTaskCreator.class)) {
                    classes.put(action, field.getAnnotation(WithTaskCreator.class).value());
                }
            } catch (NoSuchFieldException e) {
                e.printStackTrace();
            }
        }

        classes.forEach((a, c) -> taskCreatorsMap.put(
                a,
                taskCreators.stream()
                        .filter(s -> s.getClass() == c)
                        .findAny()
                        .orElseThrow(() -> new RuntimeException("Error configuring service dependencies.")))
        );
    }

    public TaskCreator get(Action action) {
        return taskCreatorsMap.getOrDefault(action, new TaskCreator() {
            @Override
            public Set<Task> create(Stage stage) {
                throw new UnsupportedOperationException("TaskCreator for " + stage + " is not defined.");
            }
        });
    }
}
