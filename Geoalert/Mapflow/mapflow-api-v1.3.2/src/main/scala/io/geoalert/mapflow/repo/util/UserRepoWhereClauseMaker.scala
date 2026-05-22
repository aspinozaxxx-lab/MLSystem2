package io.geoalert.mapflow.repo.util

import java.util.UUID

import cats.data.NonEmptyList
import doobie.Fragment
import doobie.postgres.implicits.UuidType
import doobie.util.fragment.Fragment.const
import doobie.util.fragments.in

import io.geoalert.mapflow.model.Role

object UserRepoWhereClauseMaker {
  trait WhereClauseCreator[A] {
    def whereClause(column: String, valuesForFiltering: Option[List[A]]): Option[Fragment]
  }

  implicit object BasicWhereClauseCreator extends WhereClauseCreator[String] {
    override def whereClause(
        column: String,
        valuesForFiltering: Option[List[String]],
      ): Option[Fragment] =
      valuesForFiltering.flatMap(l => NonEmptyList.fromList(l)).map(in(const(column), _))
  }

  implicit object UUIDWhereClauseCreator extends WhereClauseCreator[UUID] {
    override def whereClause(
        column: String,
        valuesForFiltering: Option[List[UUID]],
      ): Option[Fragment] =
      valuesForFiltering.flatMap(l => NonEmptyList.fromList(l)).map(in(const(column), _))
  }

  implicit object RoleWhereClauseCreator extends WhereClauseCreator[Role] {
    override def whereClause(
        column: String,
        valuesForFiltering: Option[List[Role]],
      ): Option[Fragment] =
      valuesForFiltering
        .flatMap(l => NonEmptyList.fromList(l))
        .map(rl => rl.map(_.intVal))
        .map(in(const("role"), _))
  }

  def whereClause[A](
      column: String,
      valuesForFiltering: Option[List[A]],
    )(implicit
      whereClauseCreator: WhereClauseCreator[A]
    ): Option[Fragment] =
    whereClauseCreator.whereClause(column, valuesForFiltering)
}
