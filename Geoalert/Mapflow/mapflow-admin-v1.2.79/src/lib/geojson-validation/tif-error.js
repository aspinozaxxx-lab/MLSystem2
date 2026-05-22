import { getMessageFromTiffError, getMessageFromValidation } from "./messages";

export class TiffError extends Error {
  code;
  payload;
  constructor(msg, payload) {
    super(msg.toString());
    this.code = msg;
    this.payload = payload;
  }
}

export const TiffErrorsCode = {
  LARGE_FILE: 1,
  MISSING_AFFINE_TRANSFORM: 2,
  RESOLUTION_LIMIT_EXCEEDED: 3,
  PROJECTION_ERROR: 4,
  ABORTED: 5,
};

export const parseTiffError = (error) => {
  if (error instanceof TiffError) {
    const tiffMsg = getMessageFromTiffError(error);
    const invalidMsg = getMessageFromValidation(error.code);

    return tiffMsg || invalidMsg;
  }

  return null;
};
