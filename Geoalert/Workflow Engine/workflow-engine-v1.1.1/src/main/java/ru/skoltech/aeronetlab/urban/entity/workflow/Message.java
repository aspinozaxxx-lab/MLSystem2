package ru.skoltech.aeronetlab.urban.entity.workflow;

import com.vladmihalcea.hibernate.type.basic.PostgreSQLHStoreType;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.hibernate.annotations.Type;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import java.util.Collections;
import java.util.Map;

@Embeddable
public class Message {
    @Column(length = 128)
    private String code;

    @Type(PostgreSQLHStoreType.class)
    @Column(columnDefinition = "hstore")
    private Map<String, String> parameters = Collections.emptyMap();

    @Column(length = 1024)
    private String message;

    public Message() {
    }

    public Message(String code, Map<String, String> parameters, String message) {
        this.code = StringUtils.truncate(code, 128);
        this.parameters = parameters;
        this.message = StringUtils.truncate(message, 1024);
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

    public static Message internalError(String message) {
        return new Message("workflow_engine.internalError", Collections.emptyMap(), message);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        Message that = (Message) o;

        return new EqualsBuilder()
                .append(code, that.code)
                .append(parameters, that.parameters)
                .append(message, that.message)
                .isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 37)
                .append(code)
                .append(parameters)
                .append(message)
                .toHashCode();
    }
}
