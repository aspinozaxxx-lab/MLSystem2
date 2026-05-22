import bbox from "@turf/bbox";

export function getBounds(json) {
  return bbox(JSON.parse(json));
}
