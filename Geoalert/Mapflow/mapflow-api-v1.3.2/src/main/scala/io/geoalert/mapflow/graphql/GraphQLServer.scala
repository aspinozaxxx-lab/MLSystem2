package io.geoalert.mapflow.graphql

import scala.concurrent.ExecutionContext
import scala.util.Failure
import scala.util.Success
import scala.util.Try
import scala.util.control.NonFatal

import akka.http.scaladsl.model.StatusCodes._
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.Route
import akka.stream.scaladsl.Source
import akka.util.ByteString
import cats.syntax.either._
import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe._
import io.circe.optics.JsonPath
import io.circe.optics.JsonPath._
import sangria.ast.Document
import sangria.execution._
import sangria.marshalling.circe._
import sangria.parser.QueryParser
import sangria.parser.SyntaxError

import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.graphql.schema.GraphQLSchema
import io.geoalert.mapflow.model.User

case class GraphQLRequestData(
    request: Json,
    user: User,
    map: Option[Map[String, List[String]]] = None,
    files: Map[String, Source[ByteString, Any]] = Map(),
  )

object GraphQLServer extends LazyLogging {
  val exceptionHandler: ExceptionHandler = ExceptionHandler {
    case (m, e: ApplicationError) =>
      logger.error(e.getMessage, e)
      HandledException(
        message = e.getMessage,
        additionalFields = Map("code" -> m.scalarNode(e.code, "String", Set())),
        addFieldsInExtensions = false,
        addFieldsInError = true,
      )
    case (_, e: Exception) =>
      logger.error(e.getMessage, e)
      HandledException(e.getMessage)
  }

  def endpoint(requestData: GraphQLRequestData)(implicit ec: ExecutionContext): Route = {
    val ctx = GraphQLContext(requestData.files, requestData.user)

    val request =
      mergeUploadVars(requestData.request, requestData.map.getOrElse(Map[String, List[String]]()))

    val query = root.query.string.getOption(request)
    val operation = root.operationName.string.getOption(request)
    val variables = root
      .variables
      .json
      .getOption(request)
      .map(_.asRight[ParsingFailure])
      .getOrElse(Json.obj().asRight)

    query.map(QueryParser.parse(_)) match {
      case Some(Success(ast)) =>
        variables match {
          case Left(e) => complete(formatError(e))
          case Right(js) =>
            onComplete(executeQuery(ast, operation, js, ctx)) {
              case Success(v) => ctx.authCookieDirective.getDirective(complete(v))
              case Failure(ex) => failWith(ex)
            }
        }
      case Some(Failure(e)) => complete(formatError(e))
      case None => complete((BadRequest, "No query to execute"))
    }
  }

  private def mergeUploadVars(request: Json, map: Map[String, List[String]]) = {
    def select(curPath: JsonPath, path: String) = Try(path.toInt) match {
      case Success(i) => curPath.index(i)
      case Failure(_) => curPath.selectDynamic(path)
    }
    def update(path: String, file: String)(request: Json) =
      path
        .split('.')
        .foldLeft(root)((cp, p) => select(cp, p))
        .json
        .modify(_ => Json.fromString(file))(request)
    map
      .flatMap(kv => kv._2.map(v => (kv._1, v)))
      .foldLeft(request)((r, kv) => update(kv._2, kv._1)(r))
  }

  private def formatError(error: Throwable): Json = error match {
    case syntaxError: SyntaxError =>
      Json.obj(
        "errors" -> Json.arr(
          Json.obj(
            "message" -> Json.fromString(syntaxError.getMessage),
            "locations" -> Json.arr(
              Json.obj(
                "line" -> Json.fromBigInt(syntaxError.originalError.position.line),
                "column" -> Json.fromBigInt(syntaxError.originalError.position.column),
              )
            ),
          )
        )
      )
    case NonFatal(e) => formatError(e.getMessage)
    case e => throw e
  }

  private def formatError(message: String): Json =
    Json.obj("errors" -> Json.arr(Json.obj("message" -> Json.fromString(message))))

  private def executeQuery(
      query: Document,
      operation: Option[String],
      variables: Json,
      ctx: GraphQLContext,
    )(implicit
      ec: ExecutionContext
    ) =
    Executor
      .execute(
        schema = GraphQLSchema.schema,
        queryAst = query,
        userContext = ctx,
        variables = variables,
        operationName = operation,
        exceptionHandler = exceptionHandler,
        middleware = AuthMiddleware :: Nil,
      )
      .map(OK -> _)
      .recover {
        case e: QueryAnalysisError => OK -> e.resolveError
        case e: ErrorWithResolver => OK -> e.resolveError
      }
}
