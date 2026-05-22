package io.geoalert.mapflow.implicits

import java.util.UUID

import scala.reflect.ClassTag

import cats.Applicative
import cats.Functor
import cats.Monad
import cats.data.EitherT
import cats.syntax.applicative._
import cats.syntax.either._
import cats.syntax.flatMap._
import cats.syntax.functor._
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.Forbidden
import io.geoalert.mapflow.exception.NotFound

object CommonOps {
  implicit class ApplicativeOps[A, F[_]](val value: F[List[A]]) extends AnyVal {
    def headOrNotFound(
        id: UUID
      )(implicit
        ev1: ClassTag[A],
        ev2: Applicative[F],
      ): EitherT[F, NotFound, A] =
      EitherT.fromOptionF(value.map(_.headOption), NotFound[A](id))

    def headOrForbidden(
        implicit
        ev1: ClassTag[A],
        ev2: Applicative[F],
      ): EitherT[F, ApplicationError, A] =
      EitherT.fromOptionF(value.map(_.headOption), Forbidden("No access"))
  }

  implicit class EitherTOps[F[_], A, B](val value: EitherT[F, A, B]) extends AnyVal {
    def toValidatedNec()(implicit ev: Functor[F]) = value.value.map(_.toValidatedNec)
  }

  implicit class FofEitherTOps[F[_], A, B](val value: F[Either[A, B]]) extends AnyVal {
    def toValidatedNec()(implicit ev: Functor[F]) = value.map(_.toValidatedNec)
  }

  implicit class SeqOptionToListOption[A](val value: Option[Seq[A]]) extends AnyVal {
    def listOpt: Option[List[A]] = value.map(_.toList)
  }

  implicit class SeqOps[A](val value: Seq[A]) extends AnyVal {
    def shortString(max: Int): String = {
      val elems = value.take(max).map(_.toString).mkString(", ")
      val more = value.size - max
      val andMore = if (more > 0) s" and $more more" else ""
      s"[$elems$andMore]"
    }
  }

  implicit class ListOps[A](val value: List[A]) extends AnyVal {
    def batchTraverse[F[_]: Monad, B](batchSize: Int)(f: List[A] => F[List[B]]): F[List[B]] = {
      val fs = value.grouped(batchSize).toList.map(f)
      def add(bs1: F[List[B]], bs2: F[List[B]]) = for {
        bs1 <- bs1
        bs2 <- bs2
      } yield bs1 ++ bs2
      fs.foldLeft(List[B]().pure[F])(add)
    }

    def batchTraverseSeq[F[_]: Monad, B](batchSize: Int)(f: List[A] => F[List[B]]): F[List[B]] = {
      val fs = value.grouped(batchSize).toList
      def add(bs1: F[List[B]], bs2: List[A]) = for {
        bs1 <- bs1
        bs2 <- f(bs2)
      } yield bs1 ++ bs2
      fs.foldLeft(List[B]().pure[F])(add)
    }
  }
}
