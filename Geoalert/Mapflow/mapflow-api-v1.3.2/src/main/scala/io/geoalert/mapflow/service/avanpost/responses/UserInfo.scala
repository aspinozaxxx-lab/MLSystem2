package io.geoalert.mapflow.service.avanpost.responses

import io.circe.generic.JsonCodec

@JsonCodec
case class UserInfo(
    userGroups: List[UserGroups]
  )
