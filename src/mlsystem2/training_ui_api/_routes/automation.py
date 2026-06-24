"""Automation rule routes."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._service import automation, set_automation, update_automation
from mlsystem2.training_ui_api.contracts import (
    AutomationEnabledUpdate,
    AutomationRuleInfo,
    AutomationRuleUpdate,
    AutomationSnapshot,
)

from .common import RouteContext


def register_automation_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/automation", response_model=AutomationSnapshot)
    def get_automation(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> AutomationSnapshot:
        return automation(db, ctx.config)

    @app.put("/api/v1/automation/enabled", response_model=AutomationSnapshot)
    def put_automation_enabled(
        request: AutomationEnabledUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> AutomationSnapshot:
        return set_automation(db, request, ctx.config)

    @app.put("/api/v1/automation/rules", response_model=AutomationRuleInfo)
    def put_automation_rule(
        request: AutomationRuleUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> AutomationRuleInfo:
        return update_automation(db, request, ctx.config)


__all__ = ["register_automation_routes"]
