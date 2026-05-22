package ru.skoltech.aeronetlab.urban.service.watchdog;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class WatchdogScheduler {
    @Autowired
    WatchdogService watchdogService;

    @Scheduled(fixedDelay = 60_000L)
    public void doWork() {
        watchdogService.terminateStuckWorkflows(LocalDateTime.now());
    }

}
