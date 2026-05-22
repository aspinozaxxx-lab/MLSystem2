package io.geoalert.mapflow.service.we.model

import java.time.LocalDateTime

case class Message(
    code: String,
    parameters: Map[String, String],
    message: String,
  )

case class Stage(
    id: Long,
    status: String,
    messages: Option[List[Message]],
  )

case class WorkflowResponse(
    id: Long,
    stages: List[Stage],
    status: String,
    statusUpdateDate: LocalDateTime,
  )
