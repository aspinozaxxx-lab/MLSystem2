package ru.skoltech.aeronetlab.markupstorage.exception;

import java.util.UUID;

public class FeatureCollectionNotFoundException extends RuntimeException {

    public FeatureCollectionNotFoundException(UUID id) {
        super("Feature collection with id " + id + " doesn't exist.");
    }
}
