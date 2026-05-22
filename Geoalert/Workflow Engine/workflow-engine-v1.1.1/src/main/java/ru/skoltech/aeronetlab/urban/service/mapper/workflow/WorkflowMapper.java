package ru.skoltech.aeronetlab.urban.service.mapper.workflow;

import org.locationtech.jts.geom.Geometry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.api.exception.BadRequestException;
import ru.skoltech.aeronetlab.urban.api.exception.NotFoundException;
import ru.skoltech.aeronetlab.urban.dto.api.WorkflowFilter;
import ru.skoltech.aeronetlab.urban.dto.api.WorkflowSort;
import ru.skoltech.aeronetlab.urban.dto.api.WorkflowSortEntry;
import ru.skoltech.aeronetlab.urban.dto.api.WorkflowsPage;
import ru.skoltech.aeronetlab.urban.dto.business.AreaOfInterestDto;
import ru.skoltech.aeronetlab.urban.dto.business.ArtifactDto;
import ru.skoltech.aeronetlab.urban.dto.business.RasterLayerDto;
import ru.skoltech.aeronetlab.urban.dto.business.VectorLayerDto;
import ru.skoltech.aeronetlab.urban.dto.workflow.StageDto;
import ru.skoltech.aeronetlab.urban.dto.workflow.TaskIdDto;
import ru.skoltech.aeronetlab.urban.dto.workflow.WorkflowDto;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.entity.workflow.StatusType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;
import ru.skoltech.aeronetlab.urban.entity.workflow.WorkflowStatus;

import jakarta.persistence.EntityManager;
import jakarta.persistence.TypedQuery;
import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.CriteriaQuery;
import jakarta.persistence.criteria.Expression;
import jakarta.persistence.criteria.Join;
import jakarta.persistence.criteria.Order;
import jakarta.persistence.criteria.ParameterExpression;
import jakarta.persistence.criteria.Path;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import static ru.skoltech.aeronetlab.urban.dto.api.WorkflowSortField.*;

@Service
public class WorkflowMapper {

    private static final int MAX_ENTITIES = 1_000;

    private final EntityManager entityManager;

    private final CriteriaBuilder builder;

    public WorkflowMapper(@Autowired EntityManager entityManager) {
        this.entityManager = entityManager;
        this.builder = entityManager.getCriteriaBuilder();
    }


    private TypedQuery<Long> buildQuery(WorkflowFilter filter, WorkflowSort sort,
                                        int limit, int offset, boolean count) {
        CriteriaQuery<Long> wfQuery = builder.createQuery(Long.class);

        Root<AreaOfInterest> aoi = wfQuery.from(AreaOfInterest.class);
        Join<AreaOfInterest, Workflow> workflow = aoi.join("workflow");
        Join<Workflow, WorkflowDefinitionVer> wdVer = workflow.join("workflowDefinitionVer");
        Join<WorkflowDefinitionVer, WorkflowDefinition> wd = wdVer.join("workflowDefinition");
        Join<Workflow, WorkflowStatus> status = workflow.join("workflowStatus");

        List<Predicate> conditions = new ArrayList<>();
        if (filter != null) {
            addInPredicate(conditions, workflow.get("id"), filter.getWorkflowIds());
            addInPredicate(conditions, workflow.get("system"), filter.getSystems());
            addInPredicate(conditions, workflow.get("processingId"), filter.getProcessingIds());
            addInPredicate(conditions, workflow.get("id"), filter.getWorkflowIds());
            addInPredicate(conditions, wd.get("id"), filter.getWorkflowDefinitionIds());
            if (filter.getStatuses() != null) {
                List<StatusType> statuses = filter.getStatuses().stream()
                        .map(StatusType::valueOf)
                        .collect(Collectors.toList());
                addInPredicate(conditions, status.get("status"), statuses);
            }
            if (filter.getGeometry() != null) {
                ParameterExpression<Geometry> geomParam = builder.parameter(Geometry.class, "geom");
                Expression<Boolean> intersects = builder.function(
                        "st_intersects",
                        Boolean.class,
                        aoi.get("geometry"),
                        geomParam
                );
                conditions.add(builder.isTrue(intersects));
            }
            if (filter.getCreateDateFrom() != null) {
                conditions.add(
                        builder.greaterThanOrEqualTo(workflow.get("createDate"), filter.getCreateDateFrom())
                );
            }
            if (filter.getCreateDateTo() != null) {
                conditions.add(
                        builder.lessThan(workflow.get("createDate"), filter.getCreateDateTo())
                );
            }
        }

        Function<WorkflowSortEntry, Order> sortEntryToOrder = entry -> {
            Expression<?> expr = null;
            if (entry.getSortField() == ID) {
                expr = workflow.get("id");
            } else if (entry.getSortField() == SYSTEM) {
                expr = workflow.get("system");
            } else if (entry.getSortField() == PROCESSING_ID) {
                expr = workflow.get("processingId");
            } else if (entry.getSortField() == STATUS) {
                expr = status.get("status");
            } else if (entry.getSortField() == WORKFLOW_DEFINITION_ID) {
                expr = wd.get("id");
            } else if (entry.getSortField() == CREATE_DATE) {
                expr = workflow.get("createDate");
            }

            return entry.isDesc() ? builder.desc(expr) : builder.asc(expr);
        };

        List<Order> order = new ArrayList<>();
        if (sort != null && sort.getEntries() != null) {
            order = sort.getEntries()
                    .stream()
                    .map(sortEntryToOrder)
                    .collect(Collectors.toList());
        }

        TypedQuery<Long> query = entityManager.createQuery(wfQuery
                .select(count ? builder.count(workflow.get("id")) : workflow.get("id"))
                .where(conditions.toArray(new Predicate[] {}))
                .orderBy(order)
        );

        if (offset > 0) {
            query = query.setFirstResult(offset);
        }
        if (limit > 0) {
            query = query.setMaxResults(limit);
        }

        if (filter != null && filter.getGeometry() != null) {
            query = query.setParameter("geom", filter.getGeometry());
        }

        return query;
    }

    private <T> void addInPredicate(List<Predicate> predicates, Path<Object> path, Collection<T> values) {
        if (values != null) {
            if (values.isEmpty()) {
                predicates.add(builder.or()); // always FALSE
            } else {
                predicates.add(path.in(values));
            }
        }
    }

    private List<Long> getIds(WorkflowsPage page) {
        TypedQuery<Long> query = buildQuery(
                page.getFilter(),
                page.getSort(),
                page.getLimit(),
                page.getOffset(),
                false
        );
        return query.getResultList();
    }

    public Long getCount(WorkflowFilter filter) {
        TypedQuery<Long> query = buildQuery(filter, null, 0, 0, true);
        return query.getSingleResult();
    }

    public List<WorkflowDto> getAllDtos() {
        List<Long> ids = getIds(new WorkflowsPage());
        return fetch(ids);
    }

    public List<WorkflowDto> getDtos(WorkflowsPage page) {
        return fetch(getIds(page));
    }

    public List<WorkflowDto> getDtos(List<Long> ids) {
        WorkflowFilter filter = new WorkflowFilter();
        filter.setWorkflowIds(new HashSet<>(ids));
        WorkflowsPage page = new WorkflowsPage();
        page.setFilter(filter);

        return fetch(getIds(page));
    }

    public WorkflowDto getDto(Long id) {
        WorkflowFilter filter = new WorkflowFilter();
        filter.setWorkflowIds(new HashSet<>(Collections.singletonList(id)));
        WorkflowsPage page = new WorkflowsPage();
        page.setFilter(filter);

        List<WorkflowDto> dtos = getDtos(page);
        if (dtos.isEmpty()) {
            throw new NotFoundException(WorkflowDto.class, id);
        } else {
            return dtos.get(0);
        }
    }

    private List<WorkflowDto> fetch(List<Long> ids) {
        if (ids.isEmpty()) {
            return new ArrayList<>();
        }

        if (ids.size() > MAX_ENTITIES) {
            throw new BadRequestException("Too many workflows for one request: " + ids.size());
        }

        TypedQuery<WorkflowDto> wQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.workflow.WorkflowDto(" +
                        "w.id, w.createDate, ws.updateDate, ws.status, " +
                        "wd.id, w.system, w.processingId, w.params, w.meta) " +
                        "from WorkflowStatus ws " +
                        "join ws.workflow w " +
                        "join w.workflowDefinitionVer wdv " +
                        "join wdv.workflowDefinition wd " +
                        "where w.id in :ids",
                WorkflowDto.class);
        List<WorkflowDto> workflows = wQuery.setParameter("ids", ids).getResultList();
        Map<Long, WorkflowDto> workflowsMap = new HashMap<>();
        workflows.forEach(w -> workflowsMap.put(w.getId(), w));

        TypedQuery<RasterLayerDto> rlQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.business.RasterLayerDto(" +
                        "rl.id, w.id, rl.uri) from Workflow w join w.rasterLayer rl " +
                        "where w.id in :ids",
                RasterLayerDto.class);
        List<RasterLayerDto> rasterLayers = rlQuery.setParameter("ids", ids).getResultList();
        for (RasterLayerDto rl : rasterLayers) {
            WorkflowDto workflow = workflowsMap.get(rl.getWorkflowId());
            if (workflow != null) {
                rl.setUri(rl.getUri());
                workflow.setRasterLayer(rl);
            }
        }

        TypedQuery<VectorLayerDto> vlQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.business.VectorLayerDto(" +
                        "vl.id, vl.layerId, w.id) from Workflow w join w.vectorLayer vl " +
                        "where w.id in :ids",
                VectorLayerDto.class);
        List<VectorLayerDto> vectorLayers = vlQuery.setParameter("ids", ids).getResultList();
        for (VectorLayerDto vl : vectorLayers) {
            WorkflowDto workflow = workflowsMap.get(vl.getWorkflowId());
            if (workflow != null) {
                workflow.setVectorLayer(vl);
            }
        }

        TypedQuery<AreaOfInterestDto> aoiQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.business.AreaOfInterestDto(" +
                        "a.id, a.geometry, w.id) " +
                        "from AreaOfInterest a " +
                        "join a.workflow w " +
                        "where w.id in :ids",
                AreaOfInterestDto.class);
        List<AreaOfInterestDto> aois = aoiQuery.setParameter("ids", ids).getResultList();
        for (AreaOfInterestDto aoi : aois) {
            WorkflowDto workflow = workflowsMap.get(aoi.getWorkflowId());
            if (workflow != null) {
                workflow.getAreasOfInterest().add(aoi);
            }
        }

        TypedQuery<ArtifactDto> aQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.business.ArtifactDto(" +
                        "a.areaOfInterest.id, a.artifactType, a.uri, a.workflow.id) from Artifact a " +
                        "where a.workflow.id in :ids",
                ArtifactDto.class);
        List<ArtifactDto> artifacts = aQuery.setParameter("ids", ids).getResultList();
        for (ArtifactDto a : artifacts) {
            WorkflowDto workflow = workflowsMap.get(a.getWorkflowId());
            if (workflow != null) {
                workflow.getArtifacts().add(a);
            }
        }

        TypedQuery<TaskIdDto> tQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.workflow.TaskIdDto(" +
                        "t.id, t.stage.id) " +
                        "from Task t " +
                        "where t.stage.workflow.id in :ids",
                TaskIdDto.class);
        Map<Long, List<TaskIdDto>> taskIdsByStageIds = tQuery.setParameter("ids", ids)
                .getResultStream()
                .collect(Collectors.groupingBy(TaskIdDto::getStageId));

        TypedQuery<StageDto> sQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.workflow.StageDto(" +
                        "s.id, sd.name, sd.description, " +
                        "s.workflow.id, ss) " +
                        "from StageStatus ss " +
                        "join ss.stage s " +
                        "join s.stageDefinition sd " +
                        "where ss.stage.workflow.id in :ids",
                StageDto.class);
        List<StageDto> stages = sQuery.setParameter("ids", ids).getResultList();
        for (StageDto stage : stages) {
            List<TaskIdDto> taskIds = taskIdsByStageIds.get(stage.getId());
            if (taskIds != null) {
                stage.setTaskIds(taskIds.stream().map(TaskIdDto::getTaskId).collect(Collectors.toList()));
            }

            WorkflowDto workflow = workflowsMap.get(stage.getWorkflowId());
            if (workflow != null) {
                workflow.getStages().add(stage);
            }
        }

        return ids.stream()
                .map(workflowsMap::get)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }
}