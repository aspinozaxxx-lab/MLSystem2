package ru.skoltech.aeronetlab.urban.action.selectsource;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSource;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSourceType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.repository.business.RasterSourceRepository;
import ru.skoltech.aeronetlab.urban.service.action.stagestarter.StageStarter;

import java.util.Map;
import java.util.Optional;

@Service
public class SelectSourceStageStarter implements StageStarter {

    @Autowired
    private RasterSourceRepository rasterSourceRepository;

    @Override
    public void start(Stage stage) {
        rasterSourceRepository.findByWorkflow(stage.getWorkflow())
                .ifPresent(r -> rasterSourceRepository.delete(r));
        RasterSource rasterSource = new RasterSource(stage.getWorkflow());

        copyParams(rasterSource, stage.getStageDefinition().getParams());
        copyParams(rasterSource, stage.getWorkflow().getParams());

        String sourceType = Optional.ofNullable(stage.getWorkflow().getParams().get("source_type"))
                .orElse(stage.getStageDefinition().getParams().get("source_type"));
        rasterSource.setRasterSourceType(
                RasterSourceType.fromRasterSourceName(sourceType)
                        .orElseThrow(() -> new RuntimeException(
                                "'source_type' parameter must be set to an existing raster source type value."
                        ))
        );

        rasterSource.setConfirmed(Boolean.parseBoolean(stage.getStageDefinition().getParams().get("auto_confirm")));

        rasterSourceRepository.save(rasterSource);

        if (rasterSource.isConfirmed()) {
            stage.getStageStatus().setStatus(StatusType.OK);
        }
    }

    private void copyParams(RasterSource rasterSource, Map<String, String> params) {
        copyParam(rasterSource, params, "source_type");
        copyParam(rasterSource, params, "url");
        copyParam(rasterSource, params, "zoom");
        copyParam(rasterSource, params, "projection");
        copyParam(rasterSource, params, "target_resolution");
        copyParam(rasterSource, params, "raster_login", "login");
        copyParam(rasterSource, params, "raster_password", "password");
    }

    private void copyParam(RasterSource rasterSource, Map<String, String> params, String key) {
        copyParam(rasterSource, params, key, key);
    }

    private void copyParam(RasterSource rasterSource, Map<String, String> params, String key, String targetKey) {
        Optional.ofNullable(params.get(key)).ifPresent(v -> rasterSource.getParams().put(targetKey, v));
    }
}
