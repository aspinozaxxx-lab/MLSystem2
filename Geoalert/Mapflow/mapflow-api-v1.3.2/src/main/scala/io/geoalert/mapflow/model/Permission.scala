package io.geoalert.mapflow.model

sealed abstract class Permission

object Permission {
  case object AddUser extends Permission
  case object UpdateUser extends Permission
  case object DeleteUser extends Permission
  case object ViewAnyUser extends Permission
  case object ViewOwnProject extends Permission
  case object ViewAnyProject extends Permission
  case object LargeProcessing extends Permission
  case object UnlimitedProcessing extends Permission
  case object NoZoomRestrictionsForMaxar extends Permission
  case object ManageDataProviders extends Permission
  case object ManageWorkflowDefinition extends Permission
  case object ManageTeams extends Permission
  case object ConfirmProcessingReview extends Permission
}
