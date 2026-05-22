package io.geoalert.mapflow.model

import java.util.UUID

import io.circe.generic.JsonCodec
import io.geoalert.mapflow.model.enums.MemberRole

@JsonCodec
case class UserProject(
    userId: UUID,
    projectId: UUID,
    role: MemberRole,
  )
