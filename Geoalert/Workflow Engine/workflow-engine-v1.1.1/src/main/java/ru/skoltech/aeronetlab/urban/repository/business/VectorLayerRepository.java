package ru.skoltech.aeronetlab.urban.repository.business;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.business.VectorLayer;

import java.util.Optional;
import java.util.UUID;

public interface VectorLayerRepository extends CrudRepository<VectorLayer, Long> {

    Optional<VectorLayer> findByLayerId(UUID LayerId);
}
