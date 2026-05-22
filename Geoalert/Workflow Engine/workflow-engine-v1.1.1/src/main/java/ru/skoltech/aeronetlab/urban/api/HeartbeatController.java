package ru.skoltech.aeronetlab.urban.api;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowRepository;

@RestController
@RequestMapping(path = "api/v0/heartbeat")
public class HeartbeatController {
    @Autowired
    private WorkflowRepository workflowRepository;

    /**
     * Readiness probe
     */
    @GetMapping
    public ResponseEntity<String> getHeartbeat() {
        workflowRepository.count();

        return ResponseEntity.ok("OK");
    }

    /**
     * Liveness probe
     */
    @GetMapping(path = "/lite")
    public ResponseEntity<String> getReady() {
        return ResponseEntity.ok("OK");
    }
}

