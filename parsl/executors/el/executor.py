import json
import logging
import os
import threading
import uuid
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Set, Union

import typeguard

from parsl.errors import OptionalModuleMissing
from parsl.executors.errors import InvalidResourceSpecification
from parsl.executors.status_handling import BlockProviderExecutor
from parsl.providers.base import ExecutionProvider

try:
    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
    from ensemble_launcher.config.mpi_config import MPIConfig
    from ensemble_launcher.ensemble import Task
    from ensemble_launcher.helper_functions import get_nodes
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

_LAUNCH_SCRIPT_TEMPLATE = """\
import json
import sys

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig

with open(sys.argv[1]) as f:
    sys_config = SystemConfig.model_validate(json.load(f))
with open(sys.argv[2]) as f:
    launcher_config = LauncherConfig.model_validate(json.load(f))

nodes = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None

el = EnsembleLauncher(
    ensemble_file={{}},
    system_config=sys_config,
    launcher_config=launcher_config,
    Nodes=nodes,
)
el.run()
"""


class EnsembleExecutor(BlockProviderExecutor):
    @typeguard.typechecked
    def __init__(
        self,
        cpus: List[int] = list(range(os.cpu_count())),
        gpus: List[Union[str, int]] = [],
        client_only: bool = False,
        node_id: str = "global",
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
        checkpoint_timeout: float = 300.0,
        task_buffer_size: int = 10000,
        task_flush_interval: float = 0.5,
        nodes: Optional[List[str]] = None,
        label: str = "EnsembleExecutor",
        children_scheduler_policy: str = "fixed_leafs_children_policy",
        leaf_nodes: Optional[int] = None,
        nchildren: Optional[int] = None,
        provider: Optional[ExecutionProvider] = None,
        block_error_handler: Union[bool, Callable] = True,
    ):
        if not _el_enabled:
            raise OptionalModuleMissing(
                ["ensemble_launcher"],
                "EnsembleExecutor requires the ensemble_launcher package",
            )

        super().__init__(provider=provider, block_error_handler=block_error_handler)
        self.label = label

        if provider is not None:
            if provider.nodes_per_block != 1:
                raise ValueError(
                    f"EnsembleExecutor requires provider.nodes_per_block=1, "
                    f"got {provider.nodes_per_block}"
                )
            if provider.init_blocks != 1:
                raise ValueError(
                    f"EnsembleExecutor requires provider.init_blocks=1, "
                    f"got {provider.init_blocks}"
                )
            if provider.max_blocks != 1:
                raise ValueError(
                    f"EnsembleExecutor requires provider.max_blocks=1, "
                    f"got {provider.max_blocks}"
                )

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

        if provider is None:
            self._leaf_nodes = leaf_nodes if leaf_nodes is not None else len(get_nodes())
            self._nchildren = nchildren if nchildren is not None else len(get_nodes())
        else:
            self._leaf_nodes = leaf_nodes if leaf_nodes is not None else 1
            self._nchildren = nchildren if nchildren is not None else 1

        self._checkpoint_dir_arg = checkpoint_dir
        self._n_workers = n_workers
        self._checkpoint_timeout = checkpoint_timeout
        self._task_buffer_size = task_buffer_size
        self._task_flush_interval = task_flush_interval
        self._client_only = client_only
        self._node_id = node_id
        self._nodes = nodes

        self._el: Optional[EnsembleLauncher] = None
        self._client: Optional[ClusterClient] = None
        self._checkpoint_dir: Optional[str] = None
        self._client_ready: Optional[threading.Event] = None

    def start(self) -> None:
        super().start()

        if self._checkpoint_dir_arg:
            self._checkpoint_dir = self._checkpoint_dir_arg
        else:
            self._checkpoint_dir = os.path.join(self.run_dir, self.label, "checkpoints")

        if self.provider is not None:
            self._start_via_provider()
        elif not self._client_only:
            self._start_in_process()
        else:
            self._start_client()

    def _start_via_provider(self) -> None:
        self._setup_config_files()

        self._client_ready = threading.Event()
        client_thread = threading.Thread(
            target=self._connect_client, daemon=True, name="EL-Client-Connect"
        )
        client_thread.start()

        self.initialize_scaling()

    def _start_in_process(self) -> None:
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
            policy_config=PolicyConfig(
                nlevels=self._nlevels,
                nchildren=self._nchildren,
                leaf_nodes=self._leaf_nodes,
            ),
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

        self._start_client()

    def _start_client(self) -> None:
        try:
            self._client = ClusterClient(
                checkpoint_dir=self._checkpoint_dir,
                node_id=self._node_id,
                n_workers=self._n_workers,
                checkpoint_timeout=self._checkpoint_timeout,
                task_buffer_size=self._task_buffer_size,
                task_flush_interval=self._task_flush_interval,
            )
            self._client.start()
        except Exception:
            if self._el is not None:
                self._el.stop()
                self._el = None
            raise

        logger.info("ClusterClient started with %d pipeline(s)", self._n_workers)

    def _connect_client(self) -> None:
        try:
            self._client = ClusterClient(
                checkpoint_dir=self._checkpoint_dir,
                node_id=self._node_id,
                n_workers=self._n_workers,
                checkpoint_timeout=self._checkpoint_timeout,
                task_buffer_size=self._task_buffer_size,
                task_flush_interval=self._task_flush_interval,
            )
            self._client.start()
            logger.info("ClusterClient connected to orchestrator")
        except Exception:
            logger.exception("Failed to connect ClusterClient")
        finally:
            self._client_ready.set()

    def _setup_config_files(self) -> None:
        config_dir = os.path.join(self.run_dir, self.label, "el_configs")
        os.makedirs(config_dir, exist_ok=True)

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
            policy_config=PolicyConfig(
                nlevels=self._nlevels,
                nchildren=self._nchildren,
                leaf_nodes=self._leaf_nodes,
            ),
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

        self._system_config_path = os.path.join(config_dir, "system_config.json")
        with open(self._system_config_path, "w") as f:
            f.write(sys_config.model_dump_json(indent=2))

        self._launcher_config_path = os.path.join(config_dir, "launcher_config.json")
        with open(self._launcher_config_path, "w") as f:
            f.write(launcher_config.model_dump_json(indent=2))

        self._launch_script_path = os.path.join(config_dir, "_launch_el.py")
        with open(self._launch_script_path, "w") as f:
            f.write(_LAUNCH_SCRIPT_TEMPLATE)

    def initialize_scaling(self) -> None:
        pass

    def _get_launch_command(self, block_id: str) -> str:
        cmd = f"python {self._launch_script_path} {self._system_config_path} {self._launcher_config_path}"
        if self._nodes:
            cmd += f" {','.join(self._nodes)}"
        return cmd

    def outstanding(self) -> int:
        return len(self._tasks)

    @property
    def workers_per_node(self) -> Union[int, float]:
        return 1

    @property
    def status_polling_interval(self) -> int:
        if self.provider is None:
            return 0
        return self.provider.status_polling_interval

    def submit(
        self,
        func: Callable,
        resource_specification: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        if self.bad_state_is_set:
            raise self.executor_exception

        if self._client_ready is not None:
            if not self._client_ready.wait(timeout=self._checkpoint_timeout):
                raise RuntimeError(
                    "ClusterClient failed to connect within timeout"
                )

        if self._client is None:
            raise RuntimeError("ClusterClient is not initialized")

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
        self._tasks[task_id] = fut
        fut.add_done_callback(lambda f: self._tasks.pop(task_id, None))
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

        if self.provider is not None:
            active_blocks = [
                block_id
                for block_id, status in self._status.items()
                if not status.terminal
            ]
            if active_blocks:
                self.scale_in(len(active_blocks))

        super().shutdown()

    def monitor_resources(self) -> bool:
        return False

    def _validate_resource_spec(
        self, resource_specification: Optional[Dict[str, Any]]
    ) -> None:
        pass
