package io.geoalert.mapflow

import akka.actor.ActorSystem

object AkkaSystem {
  implicit val system: ActorSystem = ActorSystem("wm-system")
}
