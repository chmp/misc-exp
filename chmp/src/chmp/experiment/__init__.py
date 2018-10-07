"""Extensions around mlflow to track experiments.
"""
import bz2
import contextlib
import json
import logging
import os.path
import tempfile
import time

from mlflow import (
    active_run,
    log_metric,
    log_artifacts,
    log_artifact,
    log_param,
    set_tag,
)


__all__ = [
    # alias standard mlflow names
    "active_run",
    "log_artifact",
    "log_artifacts",
    "log_param",
    "log_metric",
    "set_tag",
    # mlflow extensions
    "delete_experiment",
    "get_all_infos",
    "get_all_infos_df",
    "get_all_metrics",
    "get_all_params",
    "get_artifact_repository",
    "get_infos",
    "get_metric",
    "get_metric_history",
    "get_param",
    "get_store",
    "get_structured",
    "get_tempdir",
    "get_trash_folder",
    "get_run",
    "has_active_run",
    "is_local_artifact_respository",
    "list_artifacts",
    "list_all_runs",
    "log_all_metrics",
    "log_all_params",
    "log_structured",
    "log_tempdir",
    "new_run",
    "open_artifact",
    "restore_experiment",
    "set_all_tags",
]
_logger = logging.getLogger()


def get_store(uri=None):
    from mlflow import get_tracking_uri
    from mlflow.store.file_store import FileStore

    if uri is None:
        uri = get_tracking_uri()

    return FileStore(uri)


def get_artifact_repository(run_uuid, store=None):
    from mlflow.store.artifact_repo import ArtifactRepository

    store = _ensure_store(store)
    run = store.get_run(run_uuid)
    return ArtifactRepository.from_artifact_uri(run.info.artifact_uri, store)


def is_local_artifact_respository(artifact_repo):
    from mlflow.store.local_artifact_repo import LocalArtifactRepository

    return isinstance(artifact_repo, LocalArtifactRepository)


def list_all_runs(store=None):
    """Get the ids of all runs"""
    store = _ensure_store(store)

    return [
        run_info.run_uuid
        for experiment in store.list_experiments()
        for run_info in store.list_run_infos(experiment.experiment_id)
    ]


@contextlib.contextmanager
def new_run(**kwargs):
    import mlflow

    with mlflow.start_run(**kwargs) as run, tempfile.TemporaryDirectory() as tmpdir:
        _logger.info("start run %s", run.info.run_uuid)
        run.tmpdir = tmpdir

        try:
            yield run

        finally:
            log_artifacts(tmpdir)


def log_tempdir():
    run = active_run()
    assert run is not None
    log_artifacts(run.tmpdir)


def has_active_run():
    return active_run() is not None


@contextlib.contextmanager
def open_artifact(run_uuid, artifact_path, mode="rt", *, open=open, store=None):
    from mlflow.store.local_artifact_repo import LocalArtifactRepository

    artifact_repo = get_artifact_repository(run_uuid, store)
    assert is_local_artifact_respository(
        artifact_repo
    ), "only works with local artifacts"

    local_path = artifact_repo.download_artifacts(artifact_path)
    with open(local_path, mode) as fobj:
        yield fobj

    # TODO: delete local path for non local repos


def list_artifacts(run_uuid, path=".", store=None):
    artifact_repo = get_artifact_repository(run_uuid, store=store)
    return artifact_repo.list_artifacts(path)


def log_fobj(fobj, fname):
    """Log the contents as fobj as an artifact."""
    if hasattr(fobj, "encoding"):
        mode = "wt"

    else:
        mode = "wb"

    with tempfile.TemporaryDirectory() as tempdir:
        target_fname = os.path.join(tempdir, fname)
        with open(target_fname, mode) as target:
            while True:
                chunk = fobj.read(1024 * 1024)
                if len(chunk) == 0:
                    break

                target.write(chunk)

        log_artifact(target_fname)


def log_structured(**values):
    """Log a stream of structured values to an artifact inside mlflow.
    """
    tempdir = get_tempdir()

    # work around possible overflows by using strings during storage
    values = dict(values, timestamp=str(int(1000 * time.time())))

    with bz2.open(os.path.join(tempdir, "structured_log.bz2"), "at") as fobj:
        fobj.write(json.dumps(values))
        fobj.write("\n")


def get_structured(run_uuid, store=None):
    with open_artifact(
        run_uuid, "structured_log.bz2", open=bz2.open, store=store
    ) as fobj:
        # convert back the string representation of timestamps
        return [
            dict(d, timestamp=int(d["timestamp"]))
            for d in (json.loads(line) for line in fobj)
        ]


def get_tempdir():
    """Get a tempdir, after the run it will be logged as an artifact."""
    run = active_run()

    if run is None:
        raise RuntimeError("no active run")

    if not hasattr(run, "tmpdir"):
        raise RuntimeError("Run has no configured tmpdir")

    return run.tmpdir


def active_run_uuid():
    run = active_run()
    return None if run is None else run.info.run_uuid


def get_all_infos_df():
    import pandas as pd

    infos = pd.DataFrame(get_all_infos())

    infos.columns = pd.MultiIndex.from_tuples(infos.columns)
    infos["run", "start_time"] = pd.to_datetime(infos["run", "start_time"], unit="ms")
    infos["run", "end_time"] = pd.to_datetime(infos["run", "end_time"], unit="ms")
    infos["run", "duration"] = pd.to_timedelta(infos["run", "duration"], unit="ms")

    return infos


def get_all_infos(store=None):
    return [get_infos(run_uuid, store=store) for run_uuid in list_all_runs(store=store)]


def get_infos(run_uuid, store=None):
    from mlflow.entities import RunStatus

    run = get_run(run_uuid, store=store)

    if run.info.end_time is None:
        duration = None

    else:
        duration = run.info.end_time - run.info.start_time

    return {
        ("run", "uuid"): run.info.run_uuid,
        ("run", "experiment_id"): run.info.experiment_id,
        ("run", "status"): RunStatus.to_string(run.info.status),
        ("run", "start_time"): run.info.start_time,
        ("run", "end_time"): run.info.end_time,
        ("run", "duration"): duration,
        **{("metric", m.key): m.value for m in get_all_metrics(run_uuid, store=store)},
        **{("param", p.key): p.value for p in get_all_params(run_uuid, store=store)},
    }


def log_all_metrics(**metrics):
    _kv_apply(log_metric, metrics)


def log_all_params(**params):
    _kv_apply(log_param, params)


def set_all_tags(**tags):
    _kv_apply(set_tag, tags)


def _kv_apply(func, mapping):
    for k, v in mapping.items():
        func(k, v)


def get_run(run_uuid, store=None):
    return _ensure_store(store).get_run(run_uuid)


def get_all_metrics(run_uuid, store=None):
    return _ensure_store(store).get_all_metrics(run_uuid)


def get_all_params(run_uuid, store=None):
    return _ensure_store(store).get_all_params(run_uuid)


def get_metric(run_uuid, metric_key, store=None):
    return _ensure_store(store).get_metric(run_uuid, metric_key)


def get_metric_history(run_uuid, metric_key, store=None):
    return _ensure_store(store).get_metric_history(run_uuid, metric_key)


def get_param(run_uuid, param_key, store=None):
    return _ensure_store(store).get_param(run_uuid, param_key)


def delete_experiment(run_uuid, store=None):
    return _ensure_store(store).delete_experiment(run_uuid)


def restore_experiment(run_uuid, store=None):
    return _ensure_store(store).restore_experiment(store)


def get_trash_folder(store=None):
    return _ensure_store(store).trash_folder


def _ensure_store(store):
    return store if store is not None else get_store()
