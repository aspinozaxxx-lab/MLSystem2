package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.NonEmptyList
import cats.syntax.applicative._
import cats.syntax.option._
import doobie.Fragment._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment
import io.circe.Json
import io.geoalert.mapflow.DefaultDbConfig
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.implicits.Postgis._
import io.geoalert.mapflow.implicits.Postgres._

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

abstract class GenericRepo[A: Read](
    val table: String,
    val columns: Seq[String],
    val idField: String = "id",
  ) extends DefaultDbConfig {
  def listToFr[V: Put](column: String)(list: List[V]): Fragment =
    NonEmptyList.fromList(list).map(in(const(column), _)).getOrElse(fr"false")

  def valueToFr(v: Any): Fragment = v match {
    case v: String => fr"$v"
    case v: Int => fr"$v"
    case v: Long => fr"$v"
    case v: Double => fr"$v"
    case v: Boolean => fr"$v"
    case v: UUID => fr"$v"
    case v: Array[Byte] => fr"$v"
    case v: Instant => fr"$v"
    case v: Projected[Geometry] => fr"$v"
    case v: Json => fr"$v"
    case v: Map[String, String] => fr"$v"
    case v: Fragment => v
    case None => fr"null"
    case _ => sys.error(s"Bad parameter type: ${v.getClass}")
  }

  def getAllWhere(
      forUpdate: Boolean,
      where: Option[Fragment]*
    )(
      offset: Option[Int] = None,
      limit: Option[Int] = None,
      orderBy: Option[Fragment] = None,
    ): ConnectionIO[List[A]] =
    (const(s"SELECT ${columns.mkString(", ")} FROM $dbSchema.$table") ++ whereAndOpt(where: _*) ++
      (if (forUpdate) fr"FOR UPDATE" else fr"") ++
      orderBy.getOrElse(fr"") ++
      limit.map(n => const(s"LIMIT $n")).getOrElse(fr"") ++
      offset.map(n => const(s"OFFSET $n")).getOrElse(fr""))
      .query[A]
      .to[List]

  def getAllIds(forUpdate: Boolean = false): ConnectionIO[List[UUID]] =
    getAllIdsWhere(forUpdate)

  def getAllIdsWhere(forUpdate: Boolean, where: Option[Fragment]*): ConnectionIO[List[UUID]] =
    (const(s"SELECT $idField FROM $dbSchema.$table") ++ whereAndOpt(where: _*) ++ (if (forUpdate)
                                                                                     fr"FOR UPDATE"
                                                                                   else fr""))
      .query[UUID]
      .to[List]

  def getOneWhere(where: Option[Fragment]*): ConnectionIO[Option[A]] =
    getAllWhere(forUpdate = false, where: _*)().map(_.headOption)

  def getAll(forUpdate: Boolean = false): ConnectionIO[List[A]] = getAllWhere(forUpdate)()

  def getAllByIdsWhere(ids: Seq[UUID], where: Option[Fragment]*): ConnectionIO[List[A]] = {
    def getBatch(ids: List[UUID]) = NonEmptyList.fromList(ids) match {
      case Some(ids) => getAllWhere(forUpdate = false, where :+ in(const(idField), ids).some: _*)()
      case None => List[A]().pure[ConnectionIO]
    }
    ids.toList.batchTraverse(maxInFrLen)(getBatch)
  }

  def getAllByIds(ids: Seq[UUID]): ConnectionIO[List[A]] = getAllByIdsWhere(ids)

  def getOneById(id: UUID): ConnectionIO[Option[A]] =
    getAllByIds(List(id)).map(_.headOption)

  def exists(where: Option[Fragment]*): ConnectionIO[Boolean] =
    (fr"SELECT 1 FROM" ++ const(s"$dbSchema.$table") ++ whereAndOpt(where: _*) ++ const("LIMIT 1"))
      .query[Int]
      .to[List]
      .map(_.nonEmpty)

  def existsByIdWhere(id: UUID, where: Option[Fragment]*): ConnectionIO[Boolean] =
    exists(where :+ Some(const(idField) ++ fr"=$id"): _*)

  def existsById(id: UUID): ConnectionIO[Boolean] = existsByIdWhere(id)

  def nonExisting(ids: List[UUID]): ConnectionIO[List[UUID]] = {
    def batchGetExisting(ids: List[UUID]) = NonEmptyList.fromList(ids) match {
      case Some(ids) =>
        in(const(s"SELECT $idField FROM $dbSchema.$table WHERE $idField"), ids).query[UUID].to[List]
      case None => List[UUID]().pure[ConnectionIO]
    }
    for {
      existingIds <- ids.batchTraverse(maxInFrLen)(batchGetExisting)
      existing = existingIds.toSet
    } yield ids.filterNot(existing(_))
  }

  def create(columns: Map[String, Any], predefinedId: Option[UUID] = None): ConnectionIO[UUID] = {
    val id = predefinedId.getOrElse(UUID.randomUUID())
    val keys = columns.keys.filterNot(_ == idField).toSeq
    val keysFr = const(s"$idField, ${keys.mkString(", ")}")
    val valuesFr = keys.foldLeft(fr"$id")((f, k) => f ++ fr"," ++ valueToFr(columns(k)))

    val sql =
      fr"INSERT INTO" ++ const(s"$dbSchema.$table") ++ parentheses(
        keysFr
      ) ++ fr"VALUES" ++ parentheses(valuesFr)

    sql.update.run.map(_ => id)
  }

  def createOrIgnore(
      columns: Map[String, Any],
      predefinedId: Option[UUID] = None,
    ): ConnectionIO[UUID] = {
    val id = predefinedId.getOrElse(UUID.randomUUID())
    val keys = columns.keys.filterNot(_ == idField).toSeq
    val keysFr = const(s"$idField, ${keys.mkString(", ")}")
    val valuesFr = keys.foldLeft(fr"$id")((f, k) => f ++ fr"," ++ valueToFr(columns(k)))

    val sql =
      fr"INSERT INTO" ++ const(s"$dbSchema.$table") ++ parentheses(
        keysFr
      ) ++ fr"VALUES" ++ parentheses(
        valuesFr
      ) ++ fr"ON CONFLICT DO NOTHING"

    sql.update.run.map(_ => id)
  }

  def update(columns: Map[String, Any], where: Option[Fragment]*): ConnectionIO[Unit] = {
    val setFr = set(columns.toSeq.map { case (k, v) => const(s"$k=") ++ valueToFr(v) }: _*)

    (fr"UPDATE" ++ const(s"$dbSchema.$table") ++ setFr ++ whereAndOpt(where: _*))
      .update
      .run
      .map(_ => ())
  }

  def updateByIdWhere(
      columns: Map[String, Any],
      id: UUID,
      where: Option[Fragment]*
    ): ConnectionIO[Unit] =
    update(columns, where :+ Some(const(idField) ++ fr"= $id"): _*)

  def updateById(columns: Map[String, Any], id: UUID): ConnectionIO[Unit] =
    updateByIdWhere(columns, id)

  def deleteById(id: UUID): ConnectionIO[Unit] =
    (const(s"DELETE FROM $dbSchema.$table WHERE $idField") ++ fr"= $id").update.run.map(_ => ())

  def deleteByIds(ids: Seq[UUID]): ConnectionIO[Int] = {
    def batch(ids: List[UUID]) = NonEmptyList.fromList(ids) match {
      case Some(ids) =>
        (const(s"DELETE FROM $dbSchema.$table WHERE") ++ in(const(idField), ids))
          .update
          .run
          .map(List(_))
      case None => List(0).pure[ConnectionIO]
    }

    ids.toList.batchTraverse(maxInFrLen)(batch).map(_.sum)
  }

  def archive(id: UUID): ConnectionIO[Unit] =
    archive(List(id))

  def archive(ids: List[UUID]): ConnectionIO[Unit] = {
    val columns = Seq(
      ("archived" -> true).some,
      ("updated" -> Instant.now()).some,
    ).flatten.toMap

    NonEmptyList.fromList(ids) match {
      case Some(processingIds) => update(columns, in(fr"id", processingIds).some)
      case None => {}.pure[ConnectionIO]
    }
  }

  def healthCheck(): doobie.ConnectionIO[Int] =
    sql"SELECT 1".query[Int].unique
}
