package ru.skoltech.aeronetlab.urban.service.definition;

public class BadWorkflowDefinitionException extends RuntimeException {

    public BadWorkflowDefinitionException(String message) {
        super(message);
    }

    public BadWorkflowDefinitionException(String message, Throwable cause) {
        super(message, cause);
    }
}
