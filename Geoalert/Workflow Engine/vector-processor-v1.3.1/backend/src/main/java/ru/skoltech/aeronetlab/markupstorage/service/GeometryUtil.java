package ru.skoltech.aeronetlab.markupstorage.service;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.Polygon;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class GeometryUtil {

    private static Logger logger = LoggerFactory.getLogger(GeometryUtil.class);

    public static Geometry tryBuffer0(Geometry geom) {
        try {
            return geom.buffer(0);
        } catch (Exception e) {
            logger.warn("Couldn't buffer(0) geometry, because of: " + e);
            return geom;
        }
    }

    public static List<Polygon> extractPolygons(GeometryCollection geomCol) {
        List<Polygon> polygons = new ArrayList<>();

        int n = geomCol.getNumGeometries();
        for (int i = 0; i < n; i++) {
            Geometry geom = geomCol.getGeometryN(i);

            if (geom instanceof Polygon) {
                polygons.add((Polygon) geom);
            } else if (geom instanceof GeometryCollection) {
                polygons.addAll(extractPolygons((GeometryCollection) geom));
            }
        }

        return polygons;
    }
}
