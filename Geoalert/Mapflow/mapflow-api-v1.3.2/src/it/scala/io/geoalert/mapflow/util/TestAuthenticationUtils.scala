package io.geoalert.mapflow.util

import akka.http.scaladsl.model.HttpHeader
import akka.http.scaladsl.model.headers.RawHeader

import io.geoalert.mapflow.Config.apiKey
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.service.Services

object TestAuthenticationUtils extends Services {
  def authorizationHeader(user: User): HttpHeader = {
    val token = 

    RawHeader("Authorization", s"Bearer $token")
  }

  def authorizationHeaderApiKey(): HttpHeader =
    RawHeader("x-api-key", apiKey)
}
