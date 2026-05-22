package ru.skoltech.aeronetlab.urban.entity.workflow;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;

@Entity
@Table(indexes = @Index(name = "task__stage_id_idx", columnList = "stage_id"))
public class Task {

    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne
    private Stage stage;

    @ManyToOne
    private AreaOfInterest aoi;

    @OneToOne(mappedBy = "task")
    private TaskStatus taskStatus;

    @Column(columnDefinition = "TEXT")
    private String request;

    public Task() {}

    public Task(Stage stage, AreaOfInterest aoi) {
        this.stage = stage;
        this.aoi = aoi;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Stage getStage() {
        return stage;
    }

    public void setStage(Stage stage) {
        this.stage = stage;
    }

    public String getRequest() {
        return request;
    }

    public void setRequest(String request) {
        this.request = request;
    }

    public TaskStatus getTaskStatus() {
        return taskStatus;
    }

    public void setTaskStatus(TaskStatus taskStatus) {
        this.taskStatus = taskStatus;
    }

    public AreaOfInterest getAoi() {
        return aoi;
    }

    public void setAoi(AreaOfInterest aoi) {
        this.aoi = aoi;
    }

    @Override
    public String toString() {
        return String.format("Task(id=%s; stage=%s)",
                this.id,
                this.stage);
    }
}