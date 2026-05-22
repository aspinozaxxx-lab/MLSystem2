package io.geoalert.mapflow.repo.util

object RepoConstants {
  object UserRepoConstants {
    val appUserTable = "app_user"

    val idColumn = "id"
    val emailColumn = "email"
    val roleColumn = "role"
    val passwordColumn = 
    val areaLimitColumn = "area_limit"
    val aoiAreaLimitColumn = "aoi_area_limit"
    val billingTypeColumn = "billing_type"
    val createdColumn = "created"
    val updatedColumn = "updated"
    val memoryLimitColumn = "memory_limit"
    val maxAoisPerProcessingColumn = "max_aois_per_processing"
    val activeUntilColumn = "active_until"
    val reviewWorkflowEnabledColumn = "review_workflow_enabled"
    val nameColumn = "name"
    val preferredUsernameColumn = "preferred_username"
    val avantpostUserIdColumn = "avantpost_user_id"

    val appUserTableColumns: Seq[String] = Seq(
      idColumn,
      emailColumn,
      passwordColumn,
      roleColumn,
      areaLimitColumn,
      aoiAreaLimitColumn,
      billingTypeColumn,
      createdColumn,
      updatedColumn,
      memoryLimitColumn,
      maxAoisPerProcessingColumn,
      activeUntilColumn,
      reviewWorkflowEnabledColumn,
      nameColumn,
      preferredUsernameColumn,
      avantpostUserIdColumn,
    )
  }
}
