package ru.skoltech.aeronetlab.urban.service.action.statusresolver;

import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;

public interface StageStatusResolver {

    void resolve(StageStatus stageStatus);
}
