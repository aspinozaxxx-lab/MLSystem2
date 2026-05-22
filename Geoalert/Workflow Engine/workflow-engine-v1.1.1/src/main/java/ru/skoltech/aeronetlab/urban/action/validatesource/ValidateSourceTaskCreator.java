package ru.skoltech.aeronetlab.urban.action.validatesource;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.business.RasterSource;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.business.RasterSourceRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.AreaOfInterestRepository;
import ru.skoltech.aeronetlab.urban.service.action.taskcreator.TaskCreator;
import ru.skoltech.aeronetlab.urban.service.queue.TaskMessage;
import ru.skoltech.aeronetlab.urban.service.workflow.TaskService;

import java.util.*;

@Service
public class ValidateSourceTaskCreator implements TaskCreator {
    @Autowired
    private RasterSourceRepository rasterSourceRepository;

    @Autowired
    private AreaOfInterestRepository areaOfInterestRepository;

    @Autowired
    private TaskService taskService;

    @Override
    public Set<Task> create(Stage stage) {
        TaskMessage taskMessage = new TaskMessage();

        taskMessage.setInput(composeTaskInputs(stage));

        Optional<AreaOfInterest> aoiOption = areaOfInterestRepository.findByWorkflow(stage.getWorkflow());

        return Collections.singleton(taskService.create(stage, aoiOption.orElse(null), taskMessage));
    }

    private Map<String, Object> composeTaskInputs(Stage stage) {
        Map<String, Object> inputs = new HashMap<>();
        Workflow workflow = stage.getWorkflow();

        RasterSource rasterSource = rasterSourceRepository.findByWorkflow(workflow)
                .orElseThrow(() -> new RuntimeException("RasterSource for " + workflow + " doesn't exist."));

        Map<String, Object> request = new HashMap<>();

        rasterSource.getParams().entrySet().stream()
                .filter(e -> !(e.getKey().equals("login") || e.getKey().equals("password")))
                .forEach(e -> request.put(e.getKey(), e.getValue()));

        request.put("source_type", rasterSource.getRasterSourceType().getSourceTypeName());

        inputs.put("requirements", stage.getStageDefinition().getParams().get("requirements"));
        inputs.put("request", request);

        Optional<AreaOfInterest> aoiOption = areaOfInterestRepository.findByWorkflow(stage.getWorkflow());
        aoiOption.ifPresent(aoi -> inputs.put("aoi", aoi.getGeometry()));

        return inputs;

    }
}
