package ru.skoltech.aeronetlab.urban.service.definition;

import ru.skoltech.aeronetlab.urban.entity.definition.BlockConfig;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WorkflowDefinitionConfig {

    private String name;
    private Integer version;
    private Map<String, Stage> stages = new HashMap<>();

    private Collection<BlockConfig> blocks = new ArrayList<>();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getVersion() {
        return version;
    }

    public void setVersion(Integer version) {
        this.version = version;
    }

    public Map<String, Stage> getStages() {
        return stages;
    }

    public void setStages(Map<String, Stage> stages) {
        this.stages = stages;
    }

    public Collection<BlockConfig> getBlocks() {
        return blocks;
    }

    public void setBlocks(Collection<BlockConfig> blocks) {
        this.blocks = blocks;
    }

    public static class Stage {
        public String description;
        public String action;
        public List<String> dependsOn = new ArrayList<>();
        public Config config = new Config();

        public static class Config {
            public Integer retries;
            public Integer retry_interval;
            public Map<String, String> params = new HashMap<>();
        }
    }
}
