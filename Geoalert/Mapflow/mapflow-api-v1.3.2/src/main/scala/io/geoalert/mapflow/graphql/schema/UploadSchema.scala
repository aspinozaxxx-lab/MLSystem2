package io.geoalert.mapflow.graphql.schema

import sangria.schema.ScalarType
import sangria.validation.ValueCoercionViolation

import io.geoalert.mapflow.model.Upload

trait UploadSchema {
  case object UploadCoercionViolation extends ValueCoercionViolation("Multipart part name expected")

  implicit val UploadType: ScalarType[Upload] = ScalarType[Upload](
    "Upload",
    coerceOutput = (u, _) => u.part,
    coerceUserInput = {
      case part: String => Right(Upload(part))
      case _ => Left(UploadCoercionViolation)
    },
    coerceInput = {
      case sangria.ast.StringValue(part, _, _, _, _) => Right(Upload(part))
      case _ => Left(UploadCoercionViolation)
    },
  )
}
