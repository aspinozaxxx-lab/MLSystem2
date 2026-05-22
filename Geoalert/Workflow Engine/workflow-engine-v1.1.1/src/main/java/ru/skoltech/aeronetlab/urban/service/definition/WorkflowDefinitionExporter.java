package ru.skoltech.aeronetlab.urban.service.definition;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.apache.commons.lang3.tuple.Pair;
import org.springframework.core.io.InputStreamResource;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;

import java.io.ByteArrayInputStream;
import java.util.Comparator;
import java.util.stream.Collectors;

@Service
public class WorkflowDefinitionExporter {

    public InputStreamResource entityToYml(WorkflowDefinition wd) {
        WorkflowDefinitionVer lastVersion = wd.getVersions()
                .stream()
                .max(Comparator.comparingInt(WorkflowDefinitionVer::getVersion))
                .orElseThrow();

        WorkflowDefinitionConfig yml = entityToYml(lastVersion);

        ObjectMapper mapper = new ObjectMapper(new YAMLFactory()).setSerializationInclusion(JsonInclude.Include.NON_NULL);

        try {
            byte[] bytes = mapper.writeValueAsBytes(yml);
            return new InputStreamResource(new ByteArrayInputStream(bytes));
        } catch (Exception ex) {
            throw new RuntimeException("Unable to serialize object to YML for Workflow Definition " + wd.getId(), ex);
        }
    }

    private WorkflowDefinitionConfig entityToYml(WorkflowDefinitionVer entity) {
        WorkflowDefinitionConfig config = new WorkflowDefinitionConfig();
        config.setName(entity.getWorkflowDefinition().getName());
        config.setVersion(entity.getVersion());
        config.setStages(entity.getStageDefinitions().stream().map(this::entityToYml).collect(Collectors.toMap(Pair::getLeft, Pair::getRight)));
        config.setBlocks(entity.getBlockConfig());
        return config;
    }


    private Pair<String, WorkflowDefinitionConfig.Stage> entityToYml(StageDefinition definition) {
        WorkflowDefinitionConfig.Stage stage = new WorkflowDefinitionConfig.Stage();

        stage.action = definition.getAction().getActionName();
        stage.config = new WorkflowDefinitionConfig.Stage.Config();
        stage.config.params = definition.getParams();
        stage.config.retries = definition.getRetries();
        stage.config.retry_interval = definition.getRetryInterval();
        stage.description = definition.getDescription();
        stage.dependsOn = definition.getPreviousStages()
                .stream()
                .map(StageDefinition::getName)
                .collect(Collectors.toList());

        return Pair.of(definition.getName(), stage);
    }
}
