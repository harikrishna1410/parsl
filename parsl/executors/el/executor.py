import logging
import os
import uuid
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Set, Union

import typeguard

from parsl.errors import OptionalModuleMissing
from parsl.executors.base import ParslExecutor
from parsl.executors.errors import InvalidResourceSpecification

try:
    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
    from ensemble_launcher.config.mpi_config import MPIConfig
    from ensemble_launcher.ensemble import Task
    from ensemble_launcher.orchestrator import ClusterClient
except ImportError:
    _el_enabled = False
else:
    _el_enabled = True


logger = logging.getLogger(__name__)

_VALID_RESOURCE_SPEC_KEYS: Set[str] = {
    "ppn",
    "nnodes",
    "ngpus_per_process",
    "cpu_affinity",
    "gpu_affinity",
    "env",
    "run_dir",
}


class EnsembleExecutor(ParslExecutor):
    @typeguard.typechecked
    def __init__(
        self,
        cpus: List[int],
        gpus: List[Union[str, int]] = [],
        child_executor_name: str = "async_mpi",
        task_executor_name: Union[str, List[str]] = "async_processpool",
        comm_name: str = "async_zmq",
        nlevels: int = 0,
        report_interval: float = 10.0,
        return_stdout: bool = False,
        worker_logs: bool = False,
        master_logs: bool = False,
        enable_workstealing: bool = False,
        mpi_flavor: Optional[str] = None,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        overload_orchestrator_core: bool = True,
        checkpoint_dir: Optional[str] = None,
        n_workers: int = 1,
        checkpoint_timeout: float = 60.0,
        task_buffer_size: int = 10000,
        task_flush_interval: float = 0.5,
        nodes: Optional[List[str]] = None,
        label: str = "EnsembleExecutor",
    ):
        if not _el_enabled:
            raise OptionalModuleMissing(
                ["ensemble_launcher"],
                "EnsembleExecutor requires the ensemble_launcher package",
            )

        super().__init__()
        self.label = label

        self._cpus = cpus
        self._gpus = gpus

        self._child_executor_name = child_executor_name
        self._task_executor_name = task_executor_name
        self._comm_name = comm_name
        self._nlevels = nlevels
        self._report_interval = report_interval
        self._return_stdout = return_stdout
        self._worker_logs = worker_logs
        self._master_logs = master_logs
        self._enable_workstealing = enable_workstealing
        self._mpi_flavor = mpi_flavor
        self._gpu_selector = gpu_selector
        self._overload_orchestrator_core = overload_orchestrator_core

        self._checkpoint_dir_arg = checkpoint_dir
        self._n_workers = n_workers
        self._checkpoint_timeout = checkpoint_timeout
        self._task_buffer_size = task_buffer_size
        self._task_flush_interval = task_flush_interval

        self._nodes = nodes

        self._el: Optional[EnsembleLauncher] = None
        self._client: Optional[ClusterClient] = None
        self._checkpoint_dir: Optional[str] = None

    def start(self) -> None:
        if self._checkpoint_dir_arg:
            self._checkpoint_dir = self._checkpoint_dir_arg
        else:
            self._checkpoint_dir = os.path.join(self.run_dir, self.label, "checkpoints")

        sys_config = SystemConfig(
            name="parsl-el",
            cpus=self._cpus,
            gpus=self._gpus,
            ncpus=len(self._cpus),
            ngpus=len(self._gpus),
        )

        launcher_kwargs: Dict[str, Any] = dict(
            child_executor_name=self._child_executor_name,
            task_executor_name=self._task_executor_name,
            comm_name=self._comm_name,
            policy_config=PolicyConfig(nlevels=self._nlevels),
            report_interval=self._report_interval,
            return_stdout=self._return_stdout,
            worker_logs=self._worker_logs,
            master_logs=self._master_logs,
            enable_workstealing=self._enable_workstealing,
            gpu_selector=self._gpu_selector,
            overload_orchestrator_core=self._overload_orchestrator_core,
            cluster=True,
            checkpoint_dir=self._checkpoint_dir,
            log_dir=os.path.join(self.run_dir, self.label, "logs"),
        )
        if self._mpi_flavor is not None:
            launcher_kwargs["mpi_config"] = MPIConfig(flavor=self._mpi_flavor)

        launcher_config = LauncherConfig(**launcher_kwargs)

        self._el = EnsembleLauncher(
            ensemble_file={},
            system_config=sys_config,
            launcher_config=launcher_config,
            Nodes=self._nodes,
        )
        self._el.start()
        logger.info(
            "EnsembleLauncher started (checkpoint_dir=%s)", self._checkpoint_dir
        )

        try:
            self._client = ClusterClient(
                checkpoint_dir=self._checkpoint_dir,
                node_id="global",
                n_workers=self._n_workers,
                checkpoint_timeout=self._checkpoint_timeout,
                task_buffer_size=self._task_buffer_size,
                task_flush_interval=self._task_flush_interval,
            )
            self._client.start()
        except Exception:
            self._el.stop()
            self._el = None
            raise

        logger.info("ClusterClient started with %d pipeline(s)", self._n_workers)

    def submit(
        self,
        func: Callable,
        resource_specification: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        self._validate_resource_spec(resource_specification)

        res = resource_specification or {}

        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            nnodes=res.get("nnodes", 1),
            ppn=res.get("ppn", 1),
            executable=func,
            ngpus_per_process=res.get("ngpus_per_process", 0),
            args=args,
            kwargs=kwargs,
            cpu_affinity=res.get("cpu_affinity", []),
            gpu_affinity=res.get("gpu_affinity", []),
            env=res.get("env", {}),
            run_dir=res.get("run_dir"),
        )

        fut = self._client.submit(task)
        fut.parsl_executor_task_id = task_id
        return fut

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.teardown()
            except Exception:
                logger.exception("Error during ClusterClient teardown")
            self._client = None

        if self._el is not None:
            try:
                self._el.stop()
            except Exception:
                logger.exception("Error during EnsembleLauncher stop")
            self._el = None

        super().shutdown()

    def monitor_resources(self) -> bool:
        return False

    def _validate_resource_spec(
        self, resource_specification: Optional[Dict[str, Any]]
    ) -> None:
        if not resource_specification:
            return
        invalid_keys = set(resource_specification.keys()) - _VALID_RESOURCE_SPEC_KEYS
        if invalid_keys:
            raise InvalidResourceSpecification(
                invalid_keys,
                f"EnsembleExecutor accepts: {_VALID_RESOURCE_SPEC_KEYS}",
            )
