package ru.skoltech.aeronetlab.urban.api;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ru.skoltech.aeronetlab.urban.api.exception.NotFoundException;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.workflow.StageRepository;
import ru.skoltech.aeronetlab.urban.service.workflow.WorkflowService;

import java.util.Collections;
import java.util.HashSet;

@RestController
@RequestMapping(path = "api/v0/stages")
public class StageController {

    @Autowired
    private StageRepository stageRepository;

    @Autowired
    private WorkflowService workflowService;

    @PostMapping("/{stageId}/restart")
    public ResponseEntity<Void> restartStage(@PathVariable Long stageId) {
        Stage stage = stageRepository.findById(stageId)
                .orElseThrow(() -> new NotFoundException(Workflow.class, stageId));

        workflowService.restartStartingWith(new HashSet<>(Collections.singletonList(stage)));

        return ResponseEntity.ok().build();
    }
}
