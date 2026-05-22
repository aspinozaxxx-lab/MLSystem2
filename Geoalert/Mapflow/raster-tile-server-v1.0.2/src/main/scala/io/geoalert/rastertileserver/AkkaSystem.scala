package io.geoalert.rastertileserver

import akka.actor.ActorSystem

object AkkaSystem {
  implicit val system:ActorSystem = ActorSystem("rts-system")
}
