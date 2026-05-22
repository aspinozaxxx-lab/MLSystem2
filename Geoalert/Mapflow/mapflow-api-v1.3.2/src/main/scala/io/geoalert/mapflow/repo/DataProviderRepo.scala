package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.NonEmptyList
import cats.implicits.catsSyntaxApplicativeId
import cats.syntax.option._
import doobie.ConnectionIO
import doobie.Fragment._
import doobie.Fragments._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import io.geoalert.mapflow.model.CreateDataProviderInput
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.UpdateDataProviderInput

case class DataProviderDto(
    id: UUID,
    name: String,
    displayName: String,
    urlTemplate: Option[String],
    previewUrl: Option[String],
    pricePerMp: Double,
    credentialsUsername: Option[String],
    credentialsPassword: ],
    credentialsToken: ],
    isDefault: Boolean,
    archived: Boolean,
    created: Option[Instant],
    updated: Option[Instant],
    mapfileUri: Option[String],
  ) {
  def toDomain: DataProvider =
    DataProvider(
      id = id,
      name = name,
      displayName = displayName,
      urlTemplate = urlTemplate,
      previewUrl = previewUrl,
      pricePerMp = pricePerMp,
      credentialsUsername = credentialsUsername,
      credentialsPassword = ,
      credentialsToken = ,
      isDefault = isDefault,
      mapfileUri = mapfileUri,
    )
}

object DataProviderRepo
    extends GenericRepo[DataProviderDto](
      "data_provider",
      Seq(
        "id",
        "name",
        "display_name",
        "url_template",
        "preview_url",
        "price_per_mp",
        "credentials_username",
        "credentials_password",
        "credentials_token",
        "is_default",
        "archived",
        "created",
        "updated",
        "mapfile_uri",
      ),
    ) {
  def create(input: CreateDataProviderInput): ConnectionIO[UUID] = {
    val columns = Seq(
      ("name" -> input.name).some,
      ("display_name" -> input.displayName).some,
      ("url_template" -> input.urlTemplate).some,
      input.previewUrl.map("preview_url" -> _),
      ("price_per_mp" -> input.pricePerMp).some,
      input.credentialsUsername.map("credentials_username" -> _),
      input.credentialsPassword.map("credentials_password" -> _),
      input.credentialsToken.map("credentials_token" -> _),
      ("is_default" -> input.isDefault).some,
      ("created" -> Instant.now()).some,
      ("updated" -> Instant.now()).some,
      input.mapfileUri.map("mapfile_uri" -> _),
    ).flatten.toMap

    create(columns)
  }

  def update(input: UpdateDataProviderInput): ConnectionIO[Unit] = {
    val columns = Seq(
      input.name.map("name" -> _),
      input.displayName.map("display_name" -> _),
      input.urlTemplate.map("url_template" -> _),
      input.previewUrl.map("preview_url" -> _),
      input.pricePerMp.map("price_per_mp" -> _),
      input.credentialsUsername.map("credentials_username" -> _),
      input.credentialsPassword.map("credentials_password" -> _),
      input.credentialsToken.map("credentials_token" -> _),
      input.isDefault.map("is_default" -> _),
      ("updated" -> Instant.now()).some,
      input.mapfileUri.map(uri => "mapfile_uri" -> (if (uri.isBlank) None else uri.trim)),
    ).flatten.toMap

    updateById(columns, input.id)
  }

  override def getAll(forUpdate: Boolean = false): ConnectionIO[List[DataProviderDto]] =
    getAllWhere(forUpdate, fr"archived is false".some)()

  override def getAllByIds(ids: Seq[UUID]): ConnectionIO[List[DataProviderDto]] =
    getAllByIdsWhere(ids, fr"archived is false".some)

  def findByUsers(userIds: List[UUID]): ConnectionIO[Map[UUID, List[DataProvider]]] = {
    val columnsStmt = const(columns.map("dp." + _).mkString(", "));
    NonEmptyList.fromList(userIds) match {
      case Some(ids) =>
        val sql = fr"SELECT audp.user_id, " ++ columnsStmt ++
          const(
            s"FROM $dbSchema.app_user_data_provider audp JOIN $dbSchema.data_provider dp ON audp.data_provider_id = dp.id"
          ) ++
          fr"WHERE dp.archived is false AND " ++ in(fr"audp.user_id", ids)

        for {
          pairs <- sql.query[(UUID, DataProviderDto)].to[List]
          dps = pairs.map { case id -> dto => id -> dto.toDomain }.groupMap(_._1)(_._2)
        } yield dps
      case None =>
        Map[UUID, List[DataProvider]]()
          .withDefaultValue(List[DataProvider]())
          .pure[ConnectionIO]
    }
  }

  def findByUser(userId: UUID): ConnectionIO[List[DataProvider]] =
    findByUsers(List(userId))
      .map(_.getOrElse(userId, List()))

  def findDefault(): ConnectionIO[List[DataProvider]] =
    getAllWhere(false, fr"is_default is true AND archived is false".some)(None, None)
      .map(_.map(_.toDomain))

  def linkDataProvider(userId: UUID, dataProviderId: UUID): ConnectionIO[Int] = {
    val sql =
      const(s"INSERT INTO $dbSchema.app_user_data_provider (user_id, data_provider_id)") ++
        sql"VALUES ($userId, $dataProviderId) ON CONFLICT DO NOTHING"
    sql.update.run
  }

  def unlinkDataProvider(userId: UUID, dataProviderId: UUID): ConnectionIO[Int] = {
    val sql =
      const(
        s"DELETE FROM $dbSchema.app_user_data_provider"
      ) ++ sql"WHERE user_id=$userId AND data_provider_id=$dataProviderId"
    sql.update.run
  }

  def getOneByName(name: String): ConnectionIO[Option[DataProvider]] =
    getOneWhere(fr"LOWER(name) = LOWER($name)".some).map(_.map(_.toDomain))

  def listAllByNme(name: String): ConnectionIO[List[DataProvider]] =
    getAllWhere(forUpdate = false, fr"name = $name".some, fr"archived is false".some)()
      .map(_.map(_.toDomain))
}
