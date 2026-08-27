from abc import ABC, abstractmethod


class TrainingBackend(ABC):

    @abstractmethod
    def submit(self, job: dict) -> str:
        """Kick off training. Returns an external_job_id."""
        ...

    @abstractmethod
    def status(self, external_job_id: str) -> str:
        """Returns one of: 'running', 'complete', 'error'."""
        ...

    @abstractmethod
    def download(self, external_job_id: str) -> dict:
        """Once complete, retrieve results as {'final_loss':.., 'adapter_path':..}"""
        ...