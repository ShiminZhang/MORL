import argparse
from abc import ABC, abstractmethod
import logging
import os
import platform
import json
import psutil
import time
from rich.logging import RichHandler


def get_machine_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_mem = psutil.virtual_memory()
    system_info = {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    machine_info = {
        "cpu_percent": cpu_percent,
        "memory_total": virtual_mem.total,
        "memory_available": virtual_mem.available,
        "memory_percent": virtual_mem.percent,
        "memory_used": virtual_mem.used,
        "system_info": system_info,
    }
    return machine_info

class ExperimentConfig:
    def __init__(
        self,
        name,
        data_dir="data",
        result_dir="results",
        log_dir="logs",
        K=None,
        category=None,
    ):
        self.name = name
        self.data_dir = f"./Experiments/{name}/{data_dir}"
        os.makedirs(self.data_dir, exist_ok=True)
        self.result_dir = f"./Experiments/{name}/{result_dir}"
        os.makedirs(self.result_dir, exist_ok=True)
        self.log_dir = f"./Experiments/{name}/{log_dir}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.experiment_dir = f"./Experiments/{name}/"
        self.K = K
        self.category = category


class Experiment(ABC):
    _logging_configured = False

    def __init__(self, config: ExperimentConfig):
        self.metadata_path = "./experiment_metadatas.json"
        self.name = config.name
        self.config = config
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        self._configure_logging()
        self._logger_name = f"proofdoor.main.{self.name}"
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(logging.INFO)
        self.command_queue = []
        self.metadata = {
            "name": config.name,
            "data_dir": config.data_dir,
            "result_dir": config.result_dir,
            "log_dir": config.log_dir,
            "K": config.K,
            "category": config.category,
            "experiment_dir": config.experiment_dir,
        }
        self.logger.info("Experiment %s started at %s", self.name, self.start_time)


    @property
    def logger(self):
        return logging.getLogger(self._logger_name)

    def _configure_logging(self):
        if Experiment._logging_configured:
            return

        os.makedirs(self.config.log_dir, exist_ok=True)
        log_path = os.path.join(self.config.experiment_dir, f"main@{self.start_time}.log")
        if os.path.exists(log_path):
            os.remove(log_path)

        # File handler (plain text)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [pid %(process)d] %(levelname)s: %(message)s"
            )
        )

        # Console handler with Rich formatting
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            log_time_format="%H:%M:%S",
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("proofdoor")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        logger.propagate = False

        Experiment._logging_configured = True


    def get_complete_command(self, command, mem="24g", time="8:00:00", output=None, dependency=None, ntasks=None, cpus_per_task=None):
        mem_int = int(mem.split("g")[0])
        activate_python = "source ../../general/bin/activate"
        wrapped = f"{activate_python} && {command}"
        output = output if output is not None else f"{self.config.log_dir}/%j.out"
        memory = f"--mem={mem}"
        if cpus_per_task is not None and cpus_per_task > 1:
            tasks = f"--cpus-per-task={cpus_per_task}"
        elif ntasks is not None and ntasks > 1:
            tasks = f"--ntasks={ntasks}"
        else:
            tasks = ""
        dependency = f"--dependency=afterok:{dependency}" if dependency is not None else ""
        complete_command = f"sbatch {dependency} --output={output} {memory} --time={time} {tasks} --wrap=\"{wrapped}\""
        return complete_command

    def queue_command_in_slurm(self, command, mem="24g", time="16:00:00", output=None, dependency=None, ntasks=None, cpus_per_task=None):
        complete_command = self.get_complete_command(command, mem, time, output, dependency, ntasks, cpus_per_task)
        print(f"Queued command: {complete_command}")
        self.command_queue.append(complete_command)

    def execute_command_in_slurm(self, command, mem="24g", time="8:00:00", output=None, dependency=None):
        complete_command = self.get_complete_command(command, mem, time, output, dependency)
        # return the slurm id
        slurm_output =  os.popen(complete_command).read()
        return int(slurm_output.split()[-1])

    def execute_queued_command_in_slurm(self,limit=10000,batch_size=10):
        def get_queue_size():
            return int(os.popen("squeue -u $USER -h -r -t RUNNING,PENDING | wc -l").read())
        while True:
            while get_queue_size() >= limit - batch_size:
                time.sleep(300)
            print(f"Slurm queue size: {get_queue_size()}, In queue: {len(self.command_queue)}")
            for i in range(min(batch_size, len(self.command_queue))):
                command = self.command_queue.pop(0)
                os.system(command)
            if len(self.command_queue) == 0:
                break
        return

    @abstractmethod
    def experiment_main(self):
        pass

    @abstractmethod
    def on_start(self):
        pass

    @abstractmethod
    def on_end(self):
        pass

    def end(self):
        self.on_end()
        self.logger.info("Experiment %s ended", self.name)

    def execute_command_in_slurm(self, command, mem="24g", time="8:00:00", output=None):
        activate_python = "source ../general/bin/activate"
        wrapped = f"{activate_python} && {command}"
        output = output if output is not None else f"{self.config.log_dir}/%j.out"
        os.system(f"sbatch --output={output} --mem={mem} --time={time} --wrap=\"{wrapped}\"")

    def run(self):
        self.logger.info("Experiment %s started", self.name)
        metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as existing:
                try:
                    metadata = json.load(existing)
                except json.JSONDecodeError:
                    metadata = {}
        metadata[self.name] = self.metadata
        with open(self.metadata_path, "w", encoding="utf-8") as updated:
            json.dump(metadata, updated, indent=4)
        self.logger.info("Experiment config: %s", self.config)
        # log machine info
        machine_info = get_machine_info()
        self.logger.info("Machine info: %s", machine_info)
        self.on_start()
        self.experiment_main()
        return

    pass

def main():
    pass

if __name__ == "__main__":
    main()
