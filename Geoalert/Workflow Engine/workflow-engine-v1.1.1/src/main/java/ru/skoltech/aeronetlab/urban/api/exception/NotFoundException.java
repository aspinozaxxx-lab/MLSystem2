package ru.skoltech.aeronetlab.urban.api.exception;

import org.springframework.http.HttpStatus;

public class NotFoundException extends HttpException {

    public NotFoundException(Class<?> entityClass, Long id) {
        super(HttpStatus.NOT_FOUND, entityClass.getName() + " with id " + id + " doesn't exist.");
    }
}
