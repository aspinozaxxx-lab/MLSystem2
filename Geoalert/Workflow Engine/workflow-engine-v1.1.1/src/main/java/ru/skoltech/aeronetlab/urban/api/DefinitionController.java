package ru.skoltech.aeronetlab.urban.api;

import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.ResponseEntity;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import ru.skoltech.aeronetlab.urban.api.exception.BadRequestException;
import ru.skoltech.aeronetlab.urban.api.exception.NotFoundException;
import ru.skoltech.aeronetlab.urban.dto.definition.WorkflowDefinitionDto;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionRepository;
import ru.skoltech.aeronetlab.urban.service.definition.WorkflowDefinitionExporter;
import ru.skoltech.aeronetlab.urban.service.definition.WorkflowDefinitionImporter;
import ru.skoltech.aeronetlab.urban.service.mapper.definition.WorkflowDefinitionMapper;

import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

@RestController
@RequestMapping(path = "api/v0/definitions")
public class DefinitionController {
    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private WorkflowDefinitionMapper workflowDefinitionMapper;

    @Autowired
    private WorkflowDefinitionRepository workflowDefinitionRepository;

    @Autowired
    private WorkflowDefinitionImporter workflowDefinitionImporter;

    @Autowired
    private WorkflowDefinitionExporter workflowDefinitionExporter;

    @GetMapping
    public ResponseEntity<Set<WorkflowDefinitionDto>> getWorkflowDefinitions() {
        Set<WorkflowDefinitionDto> workflowDefinitions = StreamSupport
                .stream(workflowDefinitionRepository.findAll().spliterator(), false)
                .map(workflowDefinitionMapper::entityToDto)
                .collect(Collectors.toSet());

        return ResponseEntity.ok(workflowDefinitions);
    }

    @GetMapping("/{definitionId}")
    public ResponseEntity<WorkflowDefinitionDto> getWorkflowDefinition(@PathVariable Long definitionId) {
        return workflowDefinitionRepository.findById(definitionId)
                .map(workflowDefinitionMapper::entityToDto)
                .map(ResponseEntity::ok)
                .orElseThrow(() -> new NotFoundException(WorkflowDefinition.class, definitionId));
    }

    @GetMapping(value = "/{definitionId}.yml", produces = "application/x-yaml")
    public ResponseEntity<InputStreamResource> getWorkflowDefinitionYaml(@PathVariable Long definitionId) {
        return workflowDefinitionRepository.findById(definitionId)
                .map(workflowDefinitionExporter::entityToYml)
                .map(ResponseEntity::ok)
                .orElseThrow(() -> new NotFoundException(WorkflowDefinition.class, definitionId));
    }

    @PostMapping
    @Transactional
    public ResponseEntity<WorkflowDefinitionDto> importWorkflowDefinition(@RequestParam MultipartFile definition) {
        try {
            String yml = IOUtils.toString(definition.getInputStream(), StandardCharsets.UTF_8);
            log.debug("Creating WD: " + yml);
            WorkflowDefinitionVer ver = workflowDefinitionImporter.importWorkflowDefinition(yml);

            return  ResponseEntity.ok(workflowDefinitionMapper.entityToDto(ver.getWorkflowDefinition()));
        } catch (Exception e) {
            throw new BadRequestException("Error importing workflow definition: " + e);
        }
    }
}
