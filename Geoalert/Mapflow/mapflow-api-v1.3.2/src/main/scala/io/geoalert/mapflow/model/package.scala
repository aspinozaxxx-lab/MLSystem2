package io.geoalert.mapflow

package object model {
  implicit def optionString2string(maybeStr: Option[String]): String =
    maybeStr.getOrElse("-")
}
