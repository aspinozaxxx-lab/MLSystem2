package io.geoalert.mapflow.rest.json

import io.circe.Decoder
import io.circe.generic.semiauto.deriveDecoder

import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.MessageParameter
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole

import geotrellis.vector._

trait Decoders {
  // TODO: Move decoders to companion objects
  implicit val messageParameterDecoder: Decoder[MessageParameter] =
    deriveDecoder[MessageParameter]

  implicit val messageDecoder: Decoder[Message] =
    deriveDecoder[Message]

  implicit val messageJsonDecoder: Decoder[MessageJson] =
    deriveDecoder[MessageJson]

  implicit val roleDecoder: Decoder[Role] =
    Decoder.decodeString.map(Role.fromString)

  implicit val statusDecoder: Decoder[Status] =
    Decoder.decodeString.map(Status.fromString)

  implicit val aoiDecoder: Decoder[AoiJson] = deriveDecoder[AoiJson]

  implicit val teamMemberRoleDecoder: Decoder[TeamMemberRole] =
    Decoder
      .decodeString
      .map(a => TeamMemberRole.fromString(a).getOrElse(throw new IllegalArgumentException()))
}
