package ru.skoltech.aeronetlab.markupstorage.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.markupstorage.repository.FeatureRepository;

@Service
public class HeartbeatService {

    private final FeatureRepository featureRepository;

    @Autowired
    public HeartbeatService(FeatureRepository featureRepository) {
        this.featureRepository = featureRepository;
    }

    public boolean isDatabaseConnected() {
        try {
            featureRepository.existsById(1L);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
