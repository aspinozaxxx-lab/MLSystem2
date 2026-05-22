package ru.skoltech.aeronetlab.markupstorage.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.http.HttpStatus;
import ru.skoltech.aeronetlab.markupstorage.exception.FeatureCollectionNotFoundException;

@ControllerAdvice
class GlobalExceptionHandler {

    @ExceptionHandler(FeatureCollectionNotFoundException.class)
    public ResponseEntity httpException(FeatureCollectionNotFoundException ex) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(ex.getMessage());
    }

}
