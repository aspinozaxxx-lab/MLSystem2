package ru.skoltech.aeronetlab.markupstorage.dto;

import ru.skoltech.aeronetlab.markupstorage.service.merge.MergeStrategy;
import com.vividsolutions.jts.geom.Geometry;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

public class ImportParams {

    private Optional<String> layerName = Optional.empty();;
    private Optional<UUID> layerId = Optional.empty();
    private MergeStrategy mergeStrategy = MergeStrategy.INSTANCE_SEGMENTATION;
    private Optional<Geometry> mask = Optional.empty();
    private boolean cropToMask = false;
    private Set<String> keyProperties = new HashSet<>();

    public Optional<String> getLayerName() {
        return layerName;
    }

    public ImportParams setLayerName(Optional<String> layerName) {
        this.layerName = layerName;
        return this;
    }

    public Optional<UUID> getLayerId() {
        return layerId;
    }

    public ImportParams setLayerId(Optional<UUID> layerId) {
        this.layerId = layerId;
        return this;
    }

    public MergeStrategy getMergeStrategy() {
        return mergeStrategy;
    }

    public ImportParams setMergeStrategy(MergeStrategy mergeStrategy) {
        this.mergeStrategy = mergeStrategy;
        return this;
    }

    public Optional<Geometry> getMask() {
        return mask;
    }

    public ImportParams setMask(Optional<Geometry> mask) {
        this.mask = mask;
        this.mask.ifPresent(g -> g.setSRID(4326));
        return this;
    }

    public boolean isCropToMask() {
        return cropToMask;
    }

    public ImportParams setCropToMask(boolean cropToMask) {
        this.cropToMask = cropToMask;
        return this;
    }

    public Set<String> getKeyProperties() {
        return keyProperties;
    }

    public ImportParams setKeyProperties(Set<String> keyProperties) {
        this.keyProperties = keyProperties;
        return this;
    }
}
