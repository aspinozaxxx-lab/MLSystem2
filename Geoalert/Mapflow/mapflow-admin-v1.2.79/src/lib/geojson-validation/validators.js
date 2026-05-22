import distance from "@turf/distance";
import { feature, point } from "@turf/helpers";
import getKinks from "@turf/kinks";

import geojsonValidate from "geojson-validation";

import { InvalidReason } from "./types";

export class InvalidError extends Error {
  message;

  constructor(message) {
    super(message);
    this.name = "invalid-error";
    this.message = message;
  }
}

export function isGeometryObject(geojson) {
  const errors = geojsonValidate.isGeometryObject(geojson, true);
  if (Array.isArray(errors) && errors.length > 0) return errors.join(", ");
}

export function isNotEmpty(area) {
  if (area <= 0) {
    throw new InvalidError(InvalidReason.EmptyArea);
  }
}

export function maxArea(area, limit) {
  if (area > limit) {
    throw new InvalidError(InvalidReason.MaxArea);
  }
}

export function minArea(area, limit) {
  if (area < limit) {
    throw new InvalidError(InvalidReason.MinArea);
  }
}

export function rectangleSideMaxSize(coords, maxDistance) {
  const $00 = point(coords[0][0]);
  const $01 = point(coords[0][1]);
  const $02 = point(coords[0][2]);

  const xSide = distance($00, $01);
  const ySide = distance($01, $02);

  if (xSide > maxDistance || ySide > maxDistance)
    throw new InvalidError(InvalidReason.Size);
}

export function isSelfIntersection(geometry) {
  // @ts-expect-error
  const kinks = getKinks(feature(geometry));

  if (kinks.features.length) {
    throw new InvalidError(InvalidReason.SelfIntersection);
  }
}
