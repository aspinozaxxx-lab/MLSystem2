package io.geoalert.mapflow.service.avanpost.responses

import java.util.UUID

import io.circe.generic.JsonCodec

@JsonCodec
case class UserGroups(
    id: UUID,
    name: String,
  )
