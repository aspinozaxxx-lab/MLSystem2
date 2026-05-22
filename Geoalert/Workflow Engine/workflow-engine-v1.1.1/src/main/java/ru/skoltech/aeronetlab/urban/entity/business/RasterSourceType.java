package ru.skoltech.aeronetlab.urban.entity.business;

import java.util.Optional;
import java.util.stream.Stream;

public enum RasterSourceType {

    XYZ("xyz"),
    TMS("tms"),
    LOCAL("local"),
    WMS("wms"),
    QUAD_KEY("quadkey"),
    SENTINEL_L2A("sentinel_l2a"),
    NSPD("nspd"),
    HEAD_IMAGERY("head_imagery");

    private String sourceTypeName;

    public static Optional<RasterSourceType> fromRasterSourceName(String rasterSourceName) {
        return Stream.of(RasterSourceType.values())
                .filter(r -> r.getSourceTypeName().equalsIgnoreCase(rasterSourceName))
                .findAny();
    }

    RasterSourceType(String sourceTypeName) {
        this.sourceTypeName = sourceTypeName;
    }

    public String getSourceTypeName() {
        return sourceTypeName;
    }
}
