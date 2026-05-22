package ru.skoltech.aeronetlab.urban.entity.workflow;

import java.io.Serializable;

public class BlockParameters implements Serializable {
    private String name;
    private boolean enabled;

    public BlockParameters() {
    }

    public BlockParameters(String name, boolean enabled) {
        this.name = name;
        this.enabled = enabled;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}
