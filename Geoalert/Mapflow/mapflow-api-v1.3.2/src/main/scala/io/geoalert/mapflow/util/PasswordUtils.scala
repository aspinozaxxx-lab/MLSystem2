package io.geoalert.mapflow.util

import java.math.BigInteger
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

object PasswordUtils {
  def validatePassword(password: , storedHash: String): Boolean = {
    val parts = storedHash.split(":")
    val iterations = parts(0).toInt
    val salt = fromHex(parts(1))
    val hash = fromHex(parts(2))

    val spec = new PBEKeySpec(password.toCharArray, salt, iterations, hash.length * 8)
    val factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA1")
    val testHash = factory.generateSecret(spec).getEncoded

    var diff = hash.length ^ testHash.length
    for (i <- 0 until (hash.length min testHash.length))
      diff |= hash(i) ^ testHash(i)
    diff == 0
  }

  def generatePasswordHash(password:  String = {
    val iterations = 5000
    val random = new SecureRandom()
    val salt = Array.ofDim[Byte](16)
    random.nextBytes(salt)

    val spec = new PBEKeySpec(password.toCharArray, salt, iterations, 64 * 8)
    val factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA1")
    val hash = factory.generateSecret(spec).getEncoded
    s"$iterations:${toHex(salt)}:${toHex(hash)}"
  }

  private def fromHex(hex: String) = {
    val bytes = new Array[Byte](hex.length / 2)
    for (i <- bytes.indices)
      bytes(i) = Integer.parseInt(hex.substring(2 * i, 2 * i + 2), 16).toByte
    bytes
  }

  private def toHex(array: Array[Byte]) = {
    val bi = new BigInteger(1, array)
    val hex = bi.toString(16)
    val paddingLength = (array.length * 2) - hex.length
    if (paddingLength > 0) s"%0${paddingLength}d".format(0) + hex
    else hex
  }
}
