package ru.skoltech.aeronetlab.urban.entity.business;

import com.vladmihalcea.hibernate.type.basic.PostgreSQLHStoreType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import org.hibernate.annotations.Type;
import ru.skoltech.aeronetlab.urban.entity.workflow.Workflow;

import java.util.HashMap;
import java.util.Map;

@Entity
@Table(indexes = @Index(name = "raster_source__workflow_id_idx", columnList = "workflow_id"))
public class RasterSource {

    @Id
    @GeneratedValue
    private Long id;

    @OneToOne
    private Workflow workflow;

    @Enumerated(EnumType.STRING)
    private RasterSourceType rasterSourceType;

    private boolean confirmed = false;

    @Type(PostgreSQLHStoreType.class)
    @Column(columnDefinition = "hstore")
    private Map<String, String> params = new HashMap<>();

    public RasterSource() {}

    public RasterSource(Workflow workflow) {
        this.workflow = workflow;
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

    public RasterSourceType getRasterSourceType() {
        return rasterSourceType;
    }

    public void setRasterSourceType(RasterSourceType rasterSourceType) {
        this.rasterSourceType = rasterSourceType;
    }

    public boolean isConfirmed() {
        return confirmed;
    }

    public void setConfirmed(boolean confirmed) {
        this.confirmed = confirmed;
    }

    public Map<String, String> getParams() {
        return params;
    }

    @Override
    public String toString() {
        return String.format("RasterSource(id=%s; workflow=%s)",
                this.id,
                this.workflow);
    }
}
