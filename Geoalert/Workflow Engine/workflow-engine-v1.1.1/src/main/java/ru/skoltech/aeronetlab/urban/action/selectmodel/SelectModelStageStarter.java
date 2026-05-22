package ru.skoltech.aeronetlab.urban.action.selectmodel;

import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.TaskStageStarter;

@Service
public class SelectModelStageStarter extends TaskStageStarter {
    @Override
    public void start(Stage stage) {
        super.start(stage);
    }
}
