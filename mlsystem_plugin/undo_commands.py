"""Atomarnye komandy istorii review session."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from qgis.PyQt.QtGui import QUndoCommand
except ImportError:
    from qgis.PyQt.QtWidgets import QUndoCommand

if TYPE_CHECKING:
    from .geometry_splitter import SplitPart
    from .review_session import ReviewSession


class ReviewStatusCommand(QUndoCommand):
    """Menyaet kategoriyu kandidata odnoi komandoi."""

    # Fiksiruet staroe sostoyanie pered pervym redo.
    def __init__(self, session: ReviewSession, feature_id: int, new_status: str, title: str) -> None:
        super().__init__(title)
        self._session = session
        self._feature_id = feature_id
        feature = session.feature(feature_id)
        self._old_status = str(feature["review_status"] or "new")
        self._old_reviewed_at = feature["reviewed_at"]
        self._new_status = new_status

    # Primenyayet kategoriyu i perehod kak odno deistvie.
    def redo(self) -> None:
        self._session._set_review_status(self._feature_id, self._new_status)
        self._session.advance_after_action(self._feature_id)

    # Vosstanavlivaet status i tekushchii obekt.
    def undo(self) -> None:
        self._session._set_review_status(
            self._feature_id,
            self._old_status,
            reviewed_at=self._old_reviewed_at,
        )
        self._session.select_feature(self._feature_id)


class SplitCandidateCommand(QUndoCommand):
    """Dobavlyaet vse chasti i skryvaet roditelya atomarno."""

    # Fiksiruet roditelya i vse proverennye chasti.
    def __init__(
        self,
        session: ReviewSession,
        feature_id: int,
        parts: list[SplitPart],
    ) -> None:
        super().__init__("Разбить текущий объект")
        self._session = session
        self._feature_id = feature_id
        self._parts = parts
        feature = session.feature(feature_id)
        self._old_status = str(feature["review_status"] or "new")
        self._old_reviewed_at = feature["reviewed_at"]
        self._child_feature_ids: list[int] = []

    # Atomarno dobavlyaet detei i menyaet roditelya.
    def redo(self) -> None:
        self._child_feature_ids = self._session._apply_split(self._feature_id, self._parts)
        if self._child_feature_ids:
            self._session.select_feature(self._child_feature_ids[0])

    # Udalyayet vseh detei i vosstanavlivaet roditelya.
    def undo(self) -> None:
        self._session._undo_split(
            self._feature_id,
            self._child_feature_ids,
            self._old_status,
            self._old_reviewed_at,
        )
        self._session.select_feature(self._feature_id)


__all__ = ["ReviewStatusCommand", "SplitCandidateCommand"]
