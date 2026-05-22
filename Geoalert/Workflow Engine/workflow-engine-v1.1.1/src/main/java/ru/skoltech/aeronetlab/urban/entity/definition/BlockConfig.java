package ru.skoltech.aeronetlab.urban.entity.definition;

public class BlockConfig {
    private String name;

    private boolean optional;

    public BlockConfig() {
    }

    public BlockConfig(String name, boolean optional) {
        this.name = name;
        this.optional = optional;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isOptional() {
        return optional;
    }

    public void setOptional(boolean optional) {
        this.optional = optional;
    }
}
