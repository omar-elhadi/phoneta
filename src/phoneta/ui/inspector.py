"""Phoneme inspector modal — deep-dive into one word's feedback.

Shows:
* A table of reference vs user IPA phonemes with colour coding.
* A pitch-curve chart (user-only, no synthetic reference).
* Prosody summary statistics.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from phoneta.core.metrics.prosody import ProsodyResult
from phoneta.core.metrics.scoring import GREEN, RED, YELLOW, WordScore

_COLOUR_ROLE = {
    GREEN: "#2e7d32",
    YELLOW: "#f9a825",
    RED: "#c62828",
}


class InspectorDialog(QDialog):
    """Modal displaying detailed per-phoneme feedback for one word."""

    def __init__(
        self,
        word_score: WordScore,
        prosody: ProsodyResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f'Phoneme Inspector — "{word_score.word}"')
        self.resize(520, 460)
        self._build(word_score, prosody)

    def _build(self, ws: WordScore, prosody: ProsodyResult) -> None:
        layout = QVBoxLayout(self)

        # ── summary header ──────────────────────────────────────────
        header = QLabel(
            f"<b>{ws.word}</b> — {ws.accuracy:.0%} "
            f'<span style=\"color:{_COLOUR_ROLE[ws.color]}\">'
            f"({ws.color})</span>"
        )
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setWordWrap(True)
        layout.addWidget(header)

        # ── phoneme table ───────────────────────────────────────────
        if ws.feedback:
            table_group = QGroupBox("Phoneme alignment")
            tlayout = QVBoxLayout(table_group)

            tbl = QTableWidget(len(ws.feedback), 4)
            tbl.setHorizontalHeaderLabels(
                ["Reference IPA", "Your IPA", "Kind", "Confidence"]
            )
            tbl.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
            tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

            for row, fb in enumerate(ws.feedback):
                ref_item = QTableWidgetItem(fb.ref or "—")
                user_item = QTableWidgetItem(fb.user or "—")
                kind_item = QTableWidgetItem(fb.kind)
                conf_item = QTableWidgetItem(
                    f"{fb.confidence:.0%}"
                )

                if fb.flagged:
                    for item in (ref_item, user_item, kind_item, conf_item):
                        item.setForeground(Qt.GlobalColor.red)

                tbl.setItem(row, 0, ref_item)
                tbl.setItem(row, 1, user_item)
                tbl.setItem(row, 2, kind_item)
                tbl.setItem(row, 3, conf_item)

            tlayout.addWidget(tbl)
            layout.addWidget(table_group)
        else:
            layout.addWidget(QLabel("(no phoneme-level data)"))
            layout.addStretch()

        # ── prosody summary ─────────────────────────────────────────
        pgroup = QGroupBox("Prosody (your voice)")
        playout = QHBoxLayout(pgroup)
        stats = (
            f"Mean pitch: {prosody.mean_f0:.0f} Hz\n"
            f"Variability: CV = {prosody.cv_f0:.3f}\n"
            f"Voiced: {prosody.voiced_ratio:.0%}\n"
            f"Trend: {prosody.boundary_trend} "
            f"(rise: {'yes' if prosody.boundary_rise else 'no'})\n"
            f"Monotonicity: {prosody.monotonicity:.2f}"
        )
        playout.addWidget(QLabel(stats))
        layout.addWidget(pgroup)

        layout.addStretch()