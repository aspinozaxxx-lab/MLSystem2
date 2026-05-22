package ru.skoltech.aeronetlab.urban.api.exception;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

@ControllerAdvice
public class HttpExceptionHandler {

    @ExceptionHandler(HttpException.class)
    public ResponseEntity<String> httpException(HttpException ex) {
        return ResponseEntity.status(ex.getStatus()).body(ex.toString());
    }
}
