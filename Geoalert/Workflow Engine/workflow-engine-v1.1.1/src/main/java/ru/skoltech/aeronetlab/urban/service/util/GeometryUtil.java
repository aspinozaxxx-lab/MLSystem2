package ru.skoltech.aeronetlab.urban.service.util;

import geotrellis.proj4.LatLng$;
import geotrellis.proj4.WebMercator$;
import geotrellis.vector.reproject.Reproject;
import org.locationtech.jts.geom.Geometry;

public class GeometryUtil {

    public static Geometry applyBufferParam(Geometry geom, String buffer) {
        buffer = buffer.trim().toLowerCase();

        try {
            if (buffer.endsWith("m")) { // meters
                double value = Double.parseDouble(buffer.substring(0, buffer.length() - 1).trim());
                return Reproject.apply(
                        Reproject.apply(geom, LatLng$.MODULE$, WebMercator$.MODULE$).buffer(value),
                        WebMercator$.MODULE$,
                        LatLng$.MODULE$
                );
            } else if (buffer.endsWith("d")) { // degrees
                double value = Double.parseDouble(buffer.substring(0, buffer.length() - 1).trim());
                return geom.buffer(value);
            } else { // degrees
                double value = Double.parseDouble(buffer);
                return geom.buffer(value);
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Unable to parse 'buffer' param. Valid examples: 0.002, 0.002d, 200m");
        }
    }
}
