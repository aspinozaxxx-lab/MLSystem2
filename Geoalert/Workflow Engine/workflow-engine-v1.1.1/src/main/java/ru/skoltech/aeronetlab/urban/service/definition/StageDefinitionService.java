package ru.skoltech.aeronetlab.urban.service.definition;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.repository.definition.StageDefinitionRepository;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class StageDefinitionService {

    @Autowired
    private StageDefinitionRepository stageDefinitionRepository;

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    public List<StageDefinition> getTopologicalSorting(WorkflowDefinitionVer workflowDefVer) {
        Set<StageDefinition> stageDefs = stageDefinitionRepository.findAllByWorkflowDefinitionVer(workflowDefVer);
        List<StageDefinition> sorted =
                new TopologicalSorter<>(stageDefs, StageDefinition::getPreviousStages).sort();

        if (log.isDebugEnabled()) {
            String names = sorted.stream().map(StageDefinition::getName).collect(Collectors.joining(", "));
            log.debug("Topological sorting for WD: [" + names + "]");
        }

        return sorted;
    }
}
