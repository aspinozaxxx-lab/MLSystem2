package ru.skoltech.aeronetlab.urban.service.mapper.definition;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.dto.definition.WorkflowDefinitionDto;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionVerRepository;

import java.util.Comparator;

@Service
public class WorkflowDefinitionMapper {

    @Autowired
    private WorkflowDefinitionVerRepository workflowDefinitionVerRepository;

    @Autowired
    private WorkflowDefinitionVerMapper workflowDefinitionVerMapper;

    public WorkflowDefinitionDto entityToDto(WorkflowDefinition entity) {
        WorkflowDefinitionDto dto = new WorkflowDefinitionDto();

        dto.setId(entity.getId());
        dto.setName(entity.getName());

        dto.setLatestVersion(workflowDefinitionVerRepository.findLatest(entity)
                .map(workflowDefinitionVerMapper::entityToDto)
                .orElse(null));

        workflowDefinitionVerRepository.findAllByWorkflowDefinition(entity)
                .stream()
                .sorted(Comparator.comparingInt(WorkflowDefinitionVer::getVersion))
                .map(workflowDefinitionVerMapper::entityToDto)
                .forEach(v -> dto.getVersions().add(v));

        return dto;
    }
}
