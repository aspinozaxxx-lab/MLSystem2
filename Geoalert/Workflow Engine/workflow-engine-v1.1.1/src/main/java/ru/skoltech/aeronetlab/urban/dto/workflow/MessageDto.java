package ru.skoltech.aeronetlab.urban.dto.workflow;

import org.apache.commons.lang3.builder.ToStringBuilder;

import java.util.Collections;
import java.util.Map;

public class MessageDto {
    private String code;
    private Map<String, String> parameters = Collections.emptyMap();
    private String message;


    public MessageDto() {
    }

    public MessageDto(String code, Map<String, String> parameters, String message) {
        this.code = code;
        this.parameters = parameters;
        this.message = message;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public Map<String, String> getParameters() {
        return parameters;
    }

    public void setParameters(Map<String, String> parameters) {
        this.parameters = parameters;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("code", code)
                .append("parameters", parameters)
                .append("message", message)
                .toString();
    }
}
