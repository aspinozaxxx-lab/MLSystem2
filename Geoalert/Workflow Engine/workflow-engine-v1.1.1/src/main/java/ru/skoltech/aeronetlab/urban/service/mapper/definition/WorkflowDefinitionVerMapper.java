package ru.skoltech.aeronetlab.urban.service.mapper.definition;

import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.dto.definition.WorkflowDefinitionVerDto;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;

@Service
public class WorkflowDefinitionVerMapper {

    public WorkflowDefinitionVerDto entityToDto(WorkflowDefinitionVer entity) {
        WorkflowDefinitionVerDto dto = new WorkflowDefinitionVerDto();

        dto.setId(entity.getId());
        dto.setVersion(entity.getVersion());
        dto.setWorkflowDefinitionId(entity.getWorkflowDefinition().getId());

        return dto;
    }
}
