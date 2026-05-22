// import { publicRuntimeConfig } from "config";

import area from "@turf/area";
import bbox from "@turf/bbox";
import bboxPolygon from "@turf/bbox-polygon";
import cleanCoords from "@turf/clean-coords";
import { convertArea } from "@turf/helpers";

import * as validators from "./validators";
import { InvalidReason } from "./types";
import { AOI_VALIDATION } from "constants/envs";

const { MIN_AREA, MAX_AREA, MAX_SIZE } = AOI_VALIDATION;

// Validate step by step
export function validateGeometry(dirtyGeometry, options = {}) {
  try {
    const geometry = cleanCoords(dirtyGeometry);

    if (!options.skipCheckIsGeometry) {
      validators.isGeometryObject(geometry);
    }

    // Should be first in validation flow
    validators.isSelfIntersection(geometry);
    if (
      Array.isArray(geometry.coordinates[0]) &&
      geometry.coordinates[0].length > 3
    ) {
      const geometryArea = convertArea(area(geometry), "meters", "kilometers");
      validators.isNotEmpty(geometryArea);
      validators.minArea(geometryArea, MIN_AREA);
    }

    const featureBbox = bbox(geometry);
    const fbp = bboxPolygon(featureBbox);
    // const bboxArea = convertArea(area(fbp), "meters", "kilometers");
    const feautureCoords = fbp.geometry.coordinates;

    // validators.maxArea(bboxArea, MAX_AREA);

    validators.rectangleSideMaxSize(feautureCoords, MAX_SIZE);

    return { valid: true, invalidReason: null };
  } catch (error) {
    if (error instanceof validators.InvalidError) {
      return { valid: false, invalidReason: error.message };
    }

    return { valid: false, invalidReason: InvalidReason.MethodError };
  }
}
