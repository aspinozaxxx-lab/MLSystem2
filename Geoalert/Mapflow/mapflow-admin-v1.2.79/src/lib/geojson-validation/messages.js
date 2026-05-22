// import { publicRuntimeConfig } from "config";

import { t } from "@lingui/macro";
import { InvalidReason } from "./types";
import { TiffErrorsCode } from "./tif-error";
import { AOI_VALIDATION } from "constants/envs";

const {
  MIN_AREA,
  MAX_AREA,
  MAX_SIZE,
  MAX_FILE_SIZE_MB,
  MAX_IMAGE_SIZE_PIXELS,
} = AOI_VALIDATION;

export const getMessageFromValidation = (reason) => {
  switch (reason) {
    case InvalidReason.MaxArea:
      return t`Max bbox area: ${MAX_AREA} km. sq.`;
    case InvalidReason.MinArea:
      return t`Min area: ${MIN_AREA} km. sq.`;
    case InvalidReason.Size:
      return t`Max length of either bbox side: ${MAX_SIZE} km. sq.`;
    case InvalidReason.SelfIntersection:
      return t`Polygon should not has self-intersections (kinks)`;
  }

  return null;
};

export const getMessageFromTiffError = (error) => {
  switch (error.code) {
    case TiffErrorsCode.LARGE_FILE:
      return t`Too large file size, max. size: ${MAX_FILE_SIZE_MB} mb`;
    case TiffErrorsCode.MISSING_AFFINE_TRANSFORM:
      return t`Missing georeference`;
    case TiffErrorsCode.PROJECTION_ERROR:
      return t`Unsupported projection. Use web mercator, UTM or lat-lon coordinate systems`;
    case TiffErrorsCode.RESOLUTION_LIMIT_EXCEEDED:
      return t`Uploaded image width and height must be less than ${MAX_IMAGE_SIZE_PIXELS} pixels, your image has size ${error.payload}. Please cut it into parts before the upload.`;
  }

  return null;
};
