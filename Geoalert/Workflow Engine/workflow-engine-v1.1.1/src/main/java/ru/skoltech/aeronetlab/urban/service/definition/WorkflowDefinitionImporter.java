package ru.skoltech.aeronetlab.urban.service.definition;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.repository.definition.StageDefinitionRepository;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionRepository;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionVerRepository;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class WorkflowDefinitionImporter {

    @Autowired
    private WorkflowDefinitionRepository workflowDefRepository;

    @Autowired
    private WorkflowDefinitionVerRepository workflowDefVerRepository;

    @Autowired
    private StageDefinitionRepository stageDefinitionRepository;


    public WorkflowDefinitionVer importWorkflowDefinition(String yml) throws BadWorkflowDefinitionException {
        WorkflowDefinitionConfig conf = parseWorkflowDefinitionConfig(yml);

        WorkflowDefinition workflowDef = workflowDefRepository.findByName(conf.getName())
                .orElseGet(() -> createWorkflowDef(conf));

        WorkflowDefinitionVer workflowDefVer = createWorkflowDefVer(workflowDef, conf);

        Map<String, StageDefinition> stages = conf.getStages()
                .keySet()
                .stream()
                .map(stageName -> createStageDef(stageName, conf.getStages().get(stageName), workflowDefVer))
                .collect(Collectors.toMap(StageDefinition::getName, stage -> stage));

        stages.forEach((name, stage) -> resolveDependencies(stage, stages, conf));

        WorkflowDefinitionVer workflowDefVerFinal = workflowDefVerRepository.save(workflowDefVer);
        stageDefinitionRepository.saveAll(stages.values());

        return workflowDefVerFinal;
    }

    private WorkflowDefinitionConfig parseWorkflowDefinitionConfig(String yml) {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        try {
            return mapper.readValue(yml, WorkflowDefinitionConfig.class);
        } catch (IOException e) {
            throw new RuntimeException("Error parsing workflow definition: " + e.getMessage(), e);
        }
    }

    private WorkflowDefinition createWorkflowDef(WorkflowDefinitionConfig config) {
        WorkflowDefinition workflowDef = new WorkflowDefinition();
        workflowDef.setName(config.getName());

        workflowDefRepository.save(workflowDef);

        return workflowDef;
    }

    private WorkflowDefinitionVer createWorkflowDefVer(WorkflowDefinition workflowDef,
                                                       WorkflowDefinitionConfig config) {
        checkVersionNumber(workflowDef, config.getVersion());

        WorkflowDefinitionVer workflowDefVer = new WorkflowDefinitionVer();
        workflowDefVer.setWorkflowDefinition(workflowDef);
        workflowDefVer.setVersion(config.getVersion());
        workflowDefVer.setBlockConfigs(config.getBlocks());

        return workflowDefVer;
    }

    private StageDefinition createStageDef(String name, WorkflowDefinitionConfig.Stage stage,
                                           WorkflowDefinitionVer workflowDefVer) {
        StageDefinition stageDef = new StageDefinition();

        stageDef.setName(name);
        stageDef.setWorkflowDefinitionVer(workflowDefVer);
        stageDef.setDescription(stage.description);
        stageDef.setRetries(stage.config.retries);
        stageDef.setRetryInterval(stage.config.retry_interval);

        Action action = Action.fromActionName(stage.action)
                .orElseThrow(() -> new BadWorkflowDefinitionException("Action '" + stage.action + "' doesn't exist."));
        stageDef.setAction(action);

        stage.config.params.forEach((key, value) -> stageDef.getParams().put(key, value));

        return stageDef;
    }

    private void resolveDependencies(StageDefinition stageDef, Map<String, StageDefinition> stages,
                                     WorkflowDefinitionConfig config) {
        WorkflowDefinitionConfig.Stage stageConfig = config.getStages().get(stageDef.getName());

        stageConfig.dependsOn.forEach(name ->
                stageDef.getPreviousStages().add(
                        Optional.ofNullable(stages.get(name))
                                .orElseThrow(() -> new BadWorkflowDefinitionException("Stage '" + name + "' not defined."))
                )
        );
    }

    private void checkVersionNumber(WorkflowDefinition workflowDef, Integer version) {
        Optional<WorkflowDefinitionVer> latest =
                workflowDefVerRepository.findLatest(workflowDef);

        if (latest.isPresent()) {
            if (version - latest.get().getVersion() != 1) {
                throw new BadWorkflowDefinitionException("Bad workflow definition version. " +
                        "Trying to import: " + version + ". Latest present: " + latest.get().getVersion());
            }
        } else {
            if (version != 0) {
                throw new BadWorkflowDefinitionException("Bad workflow definition version. " +
                        "Currently no versions present, trying to import: " + version);
            }
        }
    }
}
