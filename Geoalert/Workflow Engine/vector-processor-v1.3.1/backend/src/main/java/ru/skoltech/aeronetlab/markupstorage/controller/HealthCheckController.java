package ru.skoltech.aeronetlab.markupstorage.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.skoltech.aeronetlab.markupstorage.service.HeartbeatService;

@RestController
@RequestMapping("/api/v0/heartbeat")
public class HealthCheckController {

    @Autowired
    private HeartbeatService heartbeatService;

    @GetMapping()
    public ResponseEntity<String> readiness() {
        if (heartbeatService.isDatabaseConnected()) {
            return ResponseEntity.ok("OK");
        } else {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body("Error");
        }
    }

    @GetMapping("/lite")
    public ResponseEntity<String> liveness() {
        return ResponseEntity.ok("OK");
    }
}
