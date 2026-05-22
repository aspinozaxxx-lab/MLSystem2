package ru.skoltech.aeronetlab.urban.service.action.stagestarter;

import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;

@Service
public class DefaultStageStarter implements StageStarter {

    @Override
    public void start(Stage stage) {}
}
