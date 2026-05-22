package ru.skoltech.aeronetlab.urban.action.selectmodel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.workflow.StageStatus;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.repository.workflow.AreaOfInterestRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.TaskRepository;
import ru.skoltech.aeronetlab.urban.service.action.statusresolver.TaskStageStatusResolver;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

@Service
public class SelectModelStageStatusResolver extends TaskStageStatusResolver {
    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private TaskRepository taskRepository;

    @Autowired
    private AreaOfInterestRepository areaOfInterestRepository;

    @Override
    public void resolve(StageStatus stageStatus) {
        log.debug("Updating stage status " + stageStatus);
        super.resolve(stageStatus);

        if (stageStatus.getStatus() == StatusType.OK) {
            Set<Task> tasks = taskRepository.findAllByStage(stageStatus.getStage());

            for (Task task : tasks) {
                Map<String, String> taskResults = task.getTaskStatus().getResults();
                AreaOfInterest aoi = task.getAoi();
                log.debug("Updating aoi params " + aoi + " " + taskResults);
                Map<String, String> params = new HashMap<>(aoi.getParams());
                params.put("model", taskResults.get("model"));
                aoi.setParams(params);
                areaOfInterestRepository.save(aoi);
            }
        }
    }
}
