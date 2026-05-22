package ru.skoltech.aeronetlab.markupstorage.service;

import com.vividsolutions.jts.geom.Geometry;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureEntity;
import ru.skoltech.aeronetlab.markupstorage.dto.Feature;

import java.util.Optional;

@Service
public class FeatureMapper {

    public FeatureEntity fromDto(Feature featureDto) {
        FeatureEntity featureEntity = new FeatureEntity();

        featureEntity.setGeometry((Geometry)featureDto.getGeometry().clone());
        Optional.ofNullable(featureDto.getProperties().get("class_id"))
                .map(this::parseClassId)
                .ifPresent(featureEntity::setClassId);
        featureEntity.getAttributes().putAll(featureDto.getProperties());
//        featureEntity.setId(featureDto.getPropertySet().getFeatureId());

        return featureEntity;
    }

    private Long parseClassId(Object classId) {
        if (classId instanceof Number) return ((Number) classId).longValue();
        else try {
            return Long.valueOf(classId.toString());
        } catch (Exception e) {
            return null;
        }
    }

    public Feature fromEntity(FeatureEntity featureEntity) {
        Feature featureDto = new Feature();

        featureDto.setGeometry((Geometry)featureEntity.getGeometry().clone());
        featureDto.getProperties().putAll(featureEntity.getAttributes());
        if (!featureDto.getProperties().containsKey("id")) {
            featureDto.getProperties().put("id", featureEntity.getId());
        }
//        featureDto.getPropertySet().setFeatureId(featureEntity.getId());

        return featureDto;
    }
}
