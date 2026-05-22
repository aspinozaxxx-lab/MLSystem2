package ru.skoltech.aeronetlab.urban.api.exception;

import org.springframework.http.HttpStatus;

public class HttpException extends RuntimeException {

    private HttpStatus status;

    public HttpException(HttpStatus status) {
        this.status = status;
    }

    public HttpException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }

    public HttpStatus getStatus() {
        return status;
    }
}
