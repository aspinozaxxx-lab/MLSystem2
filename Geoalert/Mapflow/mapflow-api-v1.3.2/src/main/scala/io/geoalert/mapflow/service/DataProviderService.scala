package io.geoalert.mapflow.service

import java.net.URL
import java.util.UUID

import scala.concurrent.duration._

import cats.data.EitherT
import cats.syntax.applicative._
import com.github.blemale.scaffeine.Cache
import com.github.blemale.scaffeine.Scaffeine
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits.AsyncConnectionIO
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model.CreateDataProviderInput
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.Permission
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.UpdateDataProviderInput
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.DataProviderRepo
import io.geoalert.mapflow.util.HttpUtils

class DataProviderService() extends LazyLogging {
  val cache: Cache[UUID, DataProvider] = Scaffeine()
    .expireAfterWrite(10.minutes)
    .maximumSize(1000)
    .build[UUID, DataProvider]()

  def listDataProviders(
      idsOpt: Option[Seq[UUID]] = None
    )(
      user: User
    ): ConnectionIO[List[DataProvider]] = {
    val either = for {
      dps <- (idsOpt, user.role) match {
        case (Some(ids), Role.Admin) =>
          EitherT.right[ApplicationError](DataProviderRepo.getAllByIds(ids).map(_.map(_.toDomain)))
        case (None, Role.User) =>
          EitherT.right[ApplicationError](for {
            defaultProviders <- DataProviderRepo.findDefault()
            userProviders <- DataProviderRepo.findByUser(user.id)
          } yield defaultProviders ++ userProviders)
        case _ => EitherT.right[ApplicationError](DataProviderRepo.getAll().map(_.map(_.toDomain)))
      }
    } yield dps.filter(dp => user.role == Role.Admin || dp.urlTemplate.isDefined)

    either.rethrowT
  }

  def createDataProvider(input: CreateDataProviderInput)(user: User): ConnectionIO[DataProvider] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.ManageDataProviders))
      id <- EitherT.right[ApplicationError](DataProviderRepo.create(input))
      providerOpt <- EitherT.right[ApplicationError](
        DataProviderRepo.getOneById(id).map(_.map(_.toDomain))
      )
    } yield providerOpt.getOrElse(throw NotFound[DataProvider](id))

    either.rethrowT
  }

  def updateDataProvider(input: UpdateDataProviderInput)(user: User): ConnectionIO[DataProvider] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.ManageDataProviders))
      _ <- EitherT.right[ApplicationError](DataProviderRepo.update(input))
      providerOpt <- EitherT.right[ApplicationError](
        DataProviderRepo.getOneById(input.id).map(_.map(_.toDomain))
      )
      _ = cache.invalidate(input.id)
    } yield providerOpt.getOrElse(throw NotFound[DataProvider](input.id))

    either.rethrowT
  }

  def getDataProvider(id: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, DataProvider] =
    cache.getIfPresent(id) match {
      case Some(value) => EitherT.rightT[ConnectionIO, ApplicationError](value)
      case None =>
        for {
          _ <- EitherT(Validations.checkDataProviderAccess(id, user))
          provider <- EitherT.right[ApplicationError](
            DataProviderRepo.getOneById(id).map(_.map(_.toDomain))
          )
          _ = if (provider.isDefined) cache.put(id, provider.get)
        } yield provider.getOrElse(throw NotFound[DataProvider](id))
    }

  def deleteDataProvider(id: UUID)(user: User): ConnectionIO[String] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.ManageDataProviders))
      _ <- EitherT.right[ApplicationError](DataProviderRepo.archive(id))
      _ = cache.invalidate(id)
    } yield "OK"

    either.rethrowT
  }

  def linkDataProvider(userId: UUID, dataProviderId: UUID)(user: User): ConnectionIO[String] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.UpdateUser))
      _ <- EitherT.right[ApplicationError](
        DataProviderRepo.linkDataProvider(userId, dataProviderId)
      )
    } yield "OK"

    either.rethrowT
  }

  def unlinkDataProvider(userId: UUID, dataProviderId: UUID)(user: User): ConnectionIO[String] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.UpdateUser))
      _ <- EitherT.right[ApplicationError](
        DataProviderRepo.unlinkDataProvider(userId, dataProviderId)
      )
    } yield "OK"

    either.rethrowT
  }

  def extractDataProviderFromUrl(urlString: String): ConnectionIO[Option[DataProvider]] = {
    val (host, connectId) =
      try {
        val url = new URL(urlString)
        val host = url.getHost
        val query = HttpUtils.parseQueryParameters(url.getQuery)

        (host, query.get("CONNECTID"))
      }
      catch {
        case _: Throwable => ("", None)
      }

    val mapboxPattern = "^(.*tiles\\.mapbox\\.com)$".r
    val worldImageryPattern = "^(.*\\.arcgisonline\\.com)$".r
    val headImageryPattern = "^(.*\\.mapflow\\.ai)$".r

    (host, connectId) match {
      case ("securewatch.digitalglobe.com", Some(connectId)) =>
        DataProviderRepo
          .getAll()
          .map(_.map(_.toDomain))
          .map(_.find(_.credentialsToken.contains(connectId)))
      case (mapboxPattern(_), _) => DataProviderRepo.getOneByName("mapbox")
      case (worldImageryPattern(_), _) => DataProviderRepo.getOneByName("arcgis_world_imagery")
      case (headImageryPattern(_), _) =>
        if (urlString.contains("/tiles/satimagery"))
          DataProviderRepo.getOneByName("head")
        else
          (None: Option[DataProvider]).pure[ConnectionIO]
      case (_, _) => (None: Option[DataProvider]).pure[ConnectionIO]
    }
  }

  def listDataProvidersByName(name: String): ConnectionIO[List[DataProvider]] =
    DataProviderRepo.listAllByNme(name)

  def retrieveDataProvider(
      params: ProcessingParams,
      urlOpt: Option[String],
    ): ConnectionIO[Option[DataProvider]] =
    params.dataProvider match {
      case Some(value) =>
        for {
          providers <- listDataProvidersByName(value)
        } yield providers.headOption

      case None => getDataProvider(urlOpt)
    }

  private def getDataProvider(url: Option[String]): ConnectionIO[Option[DataProvider]] =
    url match {
      case Some(value) => extractDataProviderFromUrl(value)
      case None => (None: Option[DataProvider]).pure[ConnectionIO]
    }
}

object DataProviderService {
  def apply(): DataProviderService = new DataProviderService()
}
