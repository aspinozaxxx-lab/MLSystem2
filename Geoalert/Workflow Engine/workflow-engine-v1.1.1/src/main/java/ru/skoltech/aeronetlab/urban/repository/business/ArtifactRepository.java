package ru.skoltech.aeronetlab.urban.repository.business;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.business.Artifact;
import ru.skoltech.aeronetlab.urban.entity.business.ArtifactType;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.util.List;
import java.util.Optional;
import java.util.Set;

public interface ArtifactRepository extends CrudRepository<Artifact, Long> {

    Set<Artifact> findAllByWorkflowAndArtifactType(Workflow workflow, ArtifactType artifactType);

    Optional<Artifact> findByWorkflowAndArtifactType(Workflow workflow, ArtifactType artifactType);

    Optional<Artifact> findByAreaOfInterestAndArtifactType(AreaOfInterest areaOfInterest, ArtifactType artifactType);

    Set<Artifact> findAllByWorkflow(Workflow workflow);

    List<Artifact> findTop5ByArtifactTypeAndCacheKeyOrderByIdDesc(ArtifactType artifactType, String cacheKey);

    boolean existsByWorkflowAndArtifactType(Workflow workflow, ArtifactType artifactType);
}
