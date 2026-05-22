package ru.skoltech.aeronetlab.urban.repository.business;

import org.springframework.data.repository.CrudRepository;
import ru.skoltech.aeronetlab.urban.entity.business.RasterLayer;

import java.util.Optional;

public interface RasterLayerRepository extends CrudRepository<RasterLayer, Long> {

    Optional<RasterLayer> findByUri(String uri);
}
