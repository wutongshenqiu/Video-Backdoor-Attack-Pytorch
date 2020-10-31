from pydantic import BaseSettings, validator

from typing import List, Optional, Union
from pathlib import PurePath


ENV_PATH = PurePath(__file__).parent / "config.env"


class Settings(BaseSettings):
    root_dir: PurePath = PurePath(__file__).parent.parent
    log_dir: PurePath = root_dir / "logs"
    source_dir: PurePath = root_dir / "src"
    model_dir: PurePath = root_dir / "models"
    trigger_dir: PurePath = root_dir / "triggers"
    logger_config_file: PurePath = source_dir / "logger_config.toml"
    dataset_dir = "~/dataset"

    test_log_path: PurePath = log_dir / "test.log"

    device: str = "cuda: 0"

    batch_size: int = 128
    num_worker: int = 4

    logger_name: str = "FileLogger"

    @validator("logger_name")
    def check_logger_name(cls, v):
        if v not in {"StreamLogger", "FileLogger"}:
            raise ValueError("unsupported logger type!")
        return v

    class Config:
        env_file = '.env'


settings = Settings(_env_file=ENV_PATH)
