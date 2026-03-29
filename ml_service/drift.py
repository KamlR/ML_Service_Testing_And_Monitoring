import asyncio
import logging
import threading
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

from ml_service import config

logger = logging.getLogger(__name__)


class DriftMonitor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []
        self._reference_data: pd.DataFrame | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    def add_event(
        self,
        features: dict[str, Any],
        prediction: int,
        probability: float,
    ) -> None:
        row = dict(features)
        row['prediction'] = prediction
        row['probability'] = probability

        with self._lock:
            self._events.append(row)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info('Drift monitor started')

    async def stop(self) -> None:
        self._running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info('Drift monitor stopped')

    async def _run_loop(self) -> None:
        interval = config.drift_check_interval_seconds()

        while self._running:
            try:
                await self._build_and_send_report_if_possible()
            except Exception:
                logger.exception('Unexpected error in drift monitoring loop')

            await asyncio.sleep(interval)

    async def _build_and_send_report_if_possible(self) -> None:
        project_id = config.evidently_project_id()
        if not project_id:
            logger.warning('EVIDENTLY_PROJECT_ID is not set, skipping drift report')
            return

        batch_size = config.drift_batch_size()

        with self._lock:
            if len(self._events) < batch_size:
                return

            batch = self._events[:batch_size]
            self._events = self._events[batch_size:]

        current_data = pd.DataFrame(batch)

        if current_data.empty:
            return

        if self._reference_data is None:
            self._reference_data = current_data.copy(deep=True)
            logger.info(
                'Reference data initialized for Evidently drift monitoring: rows=%s',
                len(self._reference_data),
            )
            return

        reference_data = self._reference_data.copy(deep=True)

        if list(reference_data.columns) != list(current_data.columns):
            common_columns = [
                column
                for column in reference_data.columns
                if column in current_data.columns
            ]
            reference_data = reference_data[common_columns]
            current_data = current_data[common_columns]

        if current_data.empty or reference_data.empty:
            logger.warning('Empty reference/current data for Evidently, skipping report')
            return

        logger.info(
            'Building Evidently drift report: reference_rows=%s current_rows=%s',
            len(reference_data),
            len(current_data),
        )

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference_data, current_data=current_data)

        workspace = RemoteWorkspace(config.evidently_url())
        workspace.add_run(project_id, result)

        logger.info('Evidently drift report uploaded successfully')

    def buffered_events_count(self) -> int:
        with self._lock:
            return len(self._events)