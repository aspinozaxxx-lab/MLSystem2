package ru.skoltech.aeronetlab.urban.entity.business;

import com.vladmihalcea.hibernate.type.basic.PostgreSQLHStoreType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import org.hibernate.annotations.Type;
import org.locationtech.jts.geom.Geometry;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.util.HashMap;
import java.util.Map;

@Entity
@Table(indexes = @Index(name = "area_of_interest__workflow_id_idx", columnList = "workflow_id"))
public class AreaOfInterest {

    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne
    private Workflow workflow;

    private Geometry geometry;

    @Type(PostgreSQLHStoreType.class)
    @Column(columnDefinition = "hstore")
    private Map<String, String> params = new HashMap<>();

    public AreaOfInterest() {}

    public AreaOfInterest(Workflow workflow, Geometry geometry) {
        this.workflow = workflow;
        this.geometry = geometry;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Workflow getWorkflow() {
        return workflow;
    }

    public void setWorkflow(Workflow workflow) {
        this.workflow = workflow;
    }

    public Geometry getGeometry() {
        return geometry;
    }

    public void setGeometry(Geometry geometry) {
        this.geometry = geometry;
    }

    public Map<String, String> getParams() {
        return params;
    }

    public void setParams(Map<String, String> params) {
        this.params = params;
    }

    @Override
    public String toString() {
        return String.format("AreaOfInterest(id=%s; workflow=%s)",
                this.id,
                this.workflow);
    }
}
