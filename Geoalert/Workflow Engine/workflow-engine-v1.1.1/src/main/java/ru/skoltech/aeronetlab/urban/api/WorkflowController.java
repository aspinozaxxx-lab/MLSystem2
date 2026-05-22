package ru.skoltech.aeronetlab.urban.api;

import org.apache.commons.lang3.StringUtils;
import org.locationtech.jts.geom.Polygonal;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import ru.skoltech.aeronetlab.urban.api.exception.BadRequestException;
import ru.skoltech.aeronetlab.urban.api.exception.NotFoundException;
import ru.skoltech.aeronetlab.urban.dto.api.WorkflowFilter;
import ru.skoltech.aeronetlab.urban.dto.api.WorkflowsPage;
import ru.skoltech.aeronetlab.urban.dto.workflow.WorkflowDto;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.repository.workflow.WorkflowRepository;
import ru.skoltech.aeronetlab.urban.service.mapper.workflow.WorkflowMapper;
import ru.skoltech.aeronetlab.urban.service.workflow.WorkflowService;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping(path = "api/v0/workflows")
public class WorkflowController {
    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private WorkflowMapper workflowMapper;

    @Autowired
    private WorkflowRepository workflowRepository;

    @Autowired
    private WorkflowService workflowService;

    @GetMapping
    public ResponseEntity<Collection<WorkflowDto>> getWorkflows(@RequestParam(required = false) List<Long> ids) {
        if (ids != null) {
            log.debug("REST GET /workflows?ids=" + StringUtils.join(ids));
            return ResponseEntity.ok(workflowMapper.getDtos(ids));
        } else {
            log.debug("REST GET /workflows");
            return ResponseEntity.ok(workflowMapper.getAllDtos());
        }
    }

    @GetMapping("/{workflowId}")
    public ResponseEntity<WorkflowDto> getWorkflow(@PathVariable Long workflowId) {
        log.debug("REST GET /workflows/" + workflowId);
        return ResponseEntity.ok(workflowMapper.getDto(workflowId));
    }

    @GetMapping("/processing/{processingId}/workflow")
    public ResponseEntity<List<WorkflowDto>> getProcessingWorkflows(@PathVariable String processingId) {
        log.debug("REST GET /processing/" + processingId + "/workflow");
        WorkflowsPage page = new WorkflowsPage();
        WorkflowFilter filter = new WorkflowFilter();
        filter.setProcessingIds(Collections.singleton(processingId));
        page.setFilter(filter);
        return ResponseEntity.ok(workflowMapper.getDtos(page));
    }

    @PostMapping("/page")
    public ResponseEntity<Collection<WorkflowDto>> getWorkflowsPage(@RequestBody WorkflowsPage page) {
        log.debug("REST POST /workflows/page Payload: " + page);
        return ResponseEntity.ok(workflowMapper.getDtos(page));
    }

    @PostMapping("/count")
    public ResponseEntity<Long> countWorkflows(@RequestBody WorkflowFilter filter) {
        log.debug("REST POST /workflows/count");
        return ResponseEntity.ok(workflowMapper.getCount(filter));
    }

    @PostMapping
    public ResponseEntity<WorkflowDto> addWorkflow(@RequestBody WorkflowDto workflow) {
        log.debug("REST POST /workflows. Payload: " + workflow);
        if (workflow.getAreasOfInterest().isEmpty()) {
            throw new BadRequestException("No area of interest provided.");
        } else if (workflow.getAreasOfInterest().size() > 1) {
            throw new BadRequestException("Only 1 area of interest per workflow is allowed.");
        } else if (!(workflow.getAreasOfInterest().get(0).getGeometry() instanceof Polygonal)) {
            throw new BadRequestException("'geometry' must be of Polygon or MultiPolygon type.");
        }

        WorkflowDto dto = workflowMapper.getDto(workflowService.create(workflow).getId());

        return ResponseEntity.ok(dto);
    }

    @PostMapping("/{workflowId}/restart")
    public ResponseEntity<Void> restartWorkflow(@PathVariable Long workflowId,
                                                @RequestParam(required = false) boolean failedStagesOnly) {
        log.debug("REST POST /workflows/" + workflowId + "/restart?failedStagesOnly=" + failedStagesOnly);
        Workflow workflow = workflowRepository.findById(workflowId)
                .orElseThrow(() -> new NotFoundException(Workflow.class, workflowId));

        if (failedStagesOnly) {
            workflowService.restartFromFailed(workflow);
        } else {
            workflowService.restart(workflow);
        }

        return ResponseEntity.ok().build();
    }

    @PostMapping("/{workflowId}/cancel")
    public ResponseEntity<Void> cancelWorkflow(@PathVariable Long workflowId) {
        log.debug("REST POST /workflows/" + workflowId + "/cancel");
        Workflow workflow = workflowRepository.findById(workflowId)
                .orElseThrow(() -> new NotFoundException(Workflow.class, workflowId));

        Optional.ofNullable(workflow.getWorkflowStatus())
                .filter(s -> s.getStatus() == StatusType.IN_PROGRESS)
                .orElseThrow(() -> new BadRequestException("Only workflow in status 'IN_PROGRESS' can be cancelled."));

        workflowService.cancel(Collections.singleton(workflowId));

        return ResponseEntity.ok().build();
    }

    @PostMapping("/batchCancel")
    public ResponseEntity<Void> cancelWorkflow(@RequestParam List<Long> workflowIds) {
        log.debug("REST POST /workflows/batchCancel Payload: " + StringUtils.join(workflowIds));

        if (workflowIds != null) {
            workflowService.cancel(new HashSet<>(workflowIds));
        }

        return ResponseEntity.ok().build();
    }
}
