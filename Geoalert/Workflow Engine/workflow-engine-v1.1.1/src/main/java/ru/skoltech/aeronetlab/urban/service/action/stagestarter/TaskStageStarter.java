package ru.skoltech.aeronetlab.urban.service.action.stagestarter;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreator;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreators;


@Service
public class TaskStageStarter implements StageStarter {

    @Autowired
    private TaskCreators taskCreators;

    @Override
    public void start(Stage stage) {
        Action action = stage.getStageDefinition().getAction();

        TaskCreator taskCreator = taskCreators.get(action);
        taskCreator.create(stage);
    }
}
