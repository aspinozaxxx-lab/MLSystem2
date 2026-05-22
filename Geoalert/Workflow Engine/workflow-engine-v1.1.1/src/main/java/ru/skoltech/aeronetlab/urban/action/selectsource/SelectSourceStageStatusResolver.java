package ru.skoltech.aeronetlab.urban.action.selectsource;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSource;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.repository.business.RasterSourceRepository;
import ru.skoltech.aeronetlab.urban.service.action.statusresolver.StageStatusResolver;

@Service
public class SelectSourceStageStatusResolver implements StageStatusResolver {

    @Autowired
    private RasterSourceRepository rasterSourceRepository;

    @Override
    public void resolve(StageStatus stageStatus) {
        Stage stage = stageStatus.getStage();
        RasterSource rasterSource = rasterSourceRepository.findByWorkflow(stage.getWorkflow())
                .orElseThrow(() -> new RuntimeException(
                        "Couldn't determine status of " + stage + " because RasterSource doesn't exist."
                ));

        if (rasterSource.isConfirmed()) {
            stageStatus.setStatus(StatusType.OK);
        } else {
            stageStatus.setStatus(StatusType.IN_PROGRESS);
        }
    }
}
