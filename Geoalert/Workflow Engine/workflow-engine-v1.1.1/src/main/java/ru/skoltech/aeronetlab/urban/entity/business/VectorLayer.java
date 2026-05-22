package ru.skoltech.aeronetlab.urban.entity.business;


import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;

import java.util.UUID;

@Entity
public class VectorLayer {

    @Id
    @GeneratedValue
    private Long id;

    @Column(columnDefinition = "uuid")
    private UUID layerId;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public UUID getLayerId() {
        return layerId;
    }

    public void setLayerId(UUID layerId) {
        this.layerId = layerId;
    }
}
