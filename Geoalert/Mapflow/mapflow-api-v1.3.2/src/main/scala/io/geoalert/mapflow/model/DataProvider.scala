package io.geoalert.mapflow.model

import java.util.UUID

case class DataProvider(
    id: UUID,
    name: String,
    displayName: String,
    urlTemplate: Option[String], /* For world wide coverage, this field contains
                         URL template. For image-based providers this field is empty */
    previewUrl: Option[String], /* For providers with world coverage,
                         we can provide previews via proxy. If not, this field is empty */
    pricePerMp: Double,
    credentialsUsername: Option[String],
    credentialsPassword: ],
    credentialsToken: ],
    isDefault: Boolean,
    mapfileUri: Option[String],
  ) {
  override def toString: String =
    s"DataProvider(\"$id\", \"$name\", \"$displayName\", \"$urlTemplate\", \"$previewUrl\", $pricePerMp, \"$credentialsUsername\", , \"[REDACTED]\", , \"$credentialsToken\", $isDefault, $mapfileUri)"

  val isMosaic: Boolean = urlTemplate.isDefined
}

case class UserCredentials(
    username: Option[String],
    password: ],
    token: ],
  )

case class CreateDataProviderInput(
    name: String,
    displayName: String,
    urlTemplate: String,
    previewUrl: Option[String],
    pricePerMp: Double,
    credentialsUsername: Option[String],
    credentialsPassword: ],
    credentialsToken: ],
    isDefault: Boolean,
    mapfileUri: Option[String],
  ) {
  override def toString: String =
    s"CreateDataProviderInput(\"$name\", \"$displayName\", \"$urlTemplate\", \"$previewUrl\", $pricePerMp, \"$credentialsUsername\", , \"[REDACTED]\", , \"$credentialsToken\", $isDefault, $mapfileUri)"
}
case class UpdateDataProviderInput(
    id: UUID,
    name: Option[String],
    displayName: Option[String],
    urlTemplate: Option[String],
    previewUrl: Option[String],
    pricePerMp: Option[Double],
    credentialsUsername: Option[String],
    credentialsPassword: ],
    credentialsToken: ],
    isDefault: Option[Boolean],
    mapfileUri: Option[String],
  ) {
  override def toString: String =
    s"DataProvider(\"$id\", \"$name\", \"$displayName\", \"$urlTemplate\", \"$previewUrl\", $pricePerMp, \"$credentialsUsername\", , \"[REDACTED]\", , \"$credentialsToken\", $isDefault, $mapfileUri)"
}
