package io.geoalert.mapflow.model

import com.typesafe.scalalogging.LazyLogging
import enumeratum.EnumEntry.UpperSnakecase
import enumeratum._
sealed trait Status extends UpperSnakecase {
  val intVal: Int
  val repr: String
}

object Status extends LazyLogging with Enum[Status] with CirceEnum[Status] {
  case object Unprocessed extends Status {
    override val intVal: Int = 0
    override val repr: String = this.entryName
  }
  case object InProgress extends Status {
    override val intVal: Int = 1
    override val repr: String = this.entryName
  }
  case object Ok extends Status {
    override val intVal: Int = 2
    override val repr: String = this.entryName
  }
  case object Failed extends Status {
    override val intVal: Int = 3
    override val repr: String = this.entryName
  }
  case object Cancelled extends Status {
    override val intVal: Int = 4
    override val repr: String = this.entryName
  }
  override def values: IndexedSeq[Status] = findValues

  lazy val statuses: List[Status] = values.toList

  def fromString(value: String): Status =
    Status
      .withNameOption(value)
      .getOrElse(throw new IllegalArgumentException(s"Unexpected status $value"))

  def fromWeStatus(weStatus: String): Status = weStatus match {
    case InProgress.repr => InProgress
    case Ok.repr => Ok
    case Failed.repr => Failed
    case Cancelled.repr => Cancelled
    case "PAUSED" => InProgress
    case _ =>
      logger.error(s"Invalid WE status: $weStatus")
      Failed
  }

  def fromProgressDetails(details: List[ProgressDetail]): Status =
    details.map(_.status).distinct match {
      case s @ _ if s.contains(InProgress) => InProgress
      case s @ _ if s.contains(Failed) => Failed
      case s @ _ if s.isEmpty => Unprocessed
      case s @ _ if s.size == 1 && s.contains(Unprocessed) => Unprocessed
      case s @ _ if s.contains(Cancelled) => Cancelled
      case _ => Ok
    }

  def apply(intVal: Int): Status = intVal match {
    case Unprocessed.intVal => Unprocessed
    case InProgress.intVal => InProgress
    case Ok.intVal => Ok
    case Failed.intVal => Failed
    case Cancelled.intVal => Cancelled
    case _ => sys.error(s"Invalid status code: $intVal")
  }
}
