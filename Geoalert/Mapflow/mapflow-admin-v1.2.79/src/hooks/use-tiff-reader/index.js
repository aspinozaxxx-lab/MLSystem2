import { useCallback, useRef, useState } from "react";
import { Tiff } from "shared/tiff";
import { convertSizeUnites } from "utils/convert-size";
import { TiffError, TiffErrorsCode } from "lib/geojson-validation/tif-error";
import { InvalidError } from "lib/geojson-validation/validators";

const IsValidFeature = { valid: true, invalidReason: null };

export const useTiffReader = (options) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState();
  const [hasPreview, setHasPreview] = useState(true);

  const [tiff, setTiff] = useState(null);

  const controller = useRef();

  const read = useCallback(
    async (files) => {
      if (loading) return;
      try {
        setLoading(true);
        const file = files[0];
        const fileSize = convertSizeUnites(file.size, "bytes", "mb");
        const maxSizeForPreview = options.maxSizeForPreview || Infinity;
        const isPreviewable = fileSize < maxSizeForPreview;

        if (options.maxSizeLimit && fileSize >= options.maxSizeLimit) {
          throw new TiffError(TiffErrorsCode.LARGE_FILE);
        }

        controller.current = new AbortController();

        const tiff = await new Tiff(file);

        const resolutionLimit = 30000;
        if (
          tiff.info.width > resolutionLimit ||
          tiff.info.height > resolutionLimit
        ) {
          throw new TiffError(
            TiffErrorsCode.RESOLUTION_LIMIT_EXCEEDED,
            (tiff.info.width > tiff.info.height
              ? tiff.info.width
              : tiff.info.height
            ).toString(),
          );
        }

        //check what i can get aoi and geoData
        try {
          //  try read geoData, error throwing inside tiff class
          const _ = tiff.geoData;
        } catch {
          throw new TiffError(TiffErrorsCode.MISSING_AFFINE_TRANSFORM);
        }

        try {
          //  try read aoi, error throwing inside tiff class
          const _ = tiff.aoi;
        } catch {
          throw new TiffError(TiffErrorsCode.PROJECTION_ERROR);
        }

        //validate
        try {
          let validRes = IsValidFeature;
          if (options?.validate) validRes = options.validate(tiff.geometry);

          if (!validRes.valid && validRes.invalidReason) {
            throw new InvalidError(validRes.invalidReason);
          }
        } catch (e) {
          if (e instanceof InvalidError) throw new TiffError(e.message); // error from validator
          throw new TiffError(TiffErrorsCode.PROJECTION_ERROR);
        }

        // Success
        // After up checks get values from cache and set in state
        setTiff(tiff);
        setHasPreview(Boolean(isPreviewable));
        if (options.onSuccess) options.onSuccess(tiff);
      } catch (error) {
        if (options.onError) options.onError(error);
        setError(error);
      } finally {
        setLoading(false);
        setError(null);
      }
    },
    [options],
  );

  const cancel = useCallback(() => {
    if (controller.current) controller.current.abort();
    setLoading(false);
    setError(null);
  }, []);

  return {
    read,
    cancel,
    error,
    isLoading: loading,
    isPreviewable: hasPreview,
    tiff,
  };
};
