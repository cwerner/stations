import itertools
import logging
import random
import sys
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from functools import reduce
from pathlib import Path
from typing import Collection

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from loguru import logger as log
from wetterdienst import DWDStationRequest, Parameter, TimeResolution

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# TODO: move this to general utility module
def formatter(record):
    lines = record["message"].splitlines()
    prefix = (
        "{time:YY-MM-DD HH:mm:ss.S} | {level.name:<8} | "
        + "{file}.{function}:{line} - ".format(**record)
    )
    indented = (
        lines[0] + "\n" + "\n".join(" " * len(prefix) + line for line in lines[1:])
    )
    record["message"] = indented.strip()
    return (
        "<g>{time:YY-MM-DD HH:mm:ss.S}</> | <lvl>{level.name:<8}</> | "
        + "<e>{file}.{function}:{line}</> - <lvl>{message}\n</>{exception}"
    )


log.remove()
log.add(sys.stderr, format=formatter)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = log.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


wetterdienst_logger = logging.getLogger("wetterdienst")
wetterdienst_logger.setLevel(1)
wetterdienst_logger.addHandler(InterceptHandler())
wetterdienst_logger.propagate = False

# silence logging from imports
# logging.getLogger("wetterdienst").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)


@dataclass
class DWDConfig:
    res: str = "hourly"
    start_date: str = "2005-01-01"
    end_date: str = "2020-06-30"
    stations: Collection[int] = field(default_factory=lambda: [2290, 2542, 5538])
    valid_ratio: float = 0.2
    sample_hours: int = 72


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=DWDConfig)


@hydra.main(config_name="config")
def main(cfg: DWDConfig) -> None:
    base_path = Path(hydra.utils.get_original_cwd())
    (base_path / "data" / "DWD" / "test").mkdir(exist_ok=True, parents=True)
    (base_path / "data" / "DWD" / "train").mkdir(exist_ok=True, parents=True)

    # hohenpeissenberg, wielenbach, altenstadt
    # target, supplier(s)
    # ids: (2290, 5538, 125)

    if cfg.res == "hourly":
        res = TimeResolution.HOURLY
    elif cfg.res == "daily":
        res = TimeResolution.DAILY
        log.error("Daily data setup valid but not implemented yet")
        raise NotImplementedError
    else:
        log.error("Only hourly time resolution allowed (for now)")
        raise NotImplementedError

    request = DWDStationRequest(
        station_ids=cfg.stations,
        parameter=[Parameter.TEMPERATURE_AIR, Parameter.PRECIPITATION],
        time_resolution=res,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        tidy_data=True,
        humanize_column_names=True,
        write_file=True,
        prefer_local=True,
    )

    dfs = []
    for df in request.collect_data():
        # sid = df.iloc[0]["STATION_ID"]
        df = df[["ELEMENT", "DATE", "VALUE"]]
        df = df.pivot(index="DATE", columns="ELEMENT", values="VALUE")
        dfs.append(df[["PRECIPITATION_HEIGHT", "TEMPERATURE_AIR_200"]])

    # merge and keep indices
    df = reduce(
        lambda df_left, df_right: pd.merge(
            df_left, df_right, left_index=True, right_index=True, how="outer"
        ),
        dfs,
    )

    target_cols = ["TARGET_PRCP", "TARGET_TEMP"]
    source_cols = [[f"SRC{d:02d}_PRCP", f"SRC{d:02d}_TEMP"] for d in range(1, len(dfs))]
    df.columns = target_cols + list(itertools.chain(*source_cols))

    log.debug(f"Sample:\n{df.head()}")

    n = len(df)
    rows = range(n)
    row_idx = sorted(random.choices(rows, k=n // 3))

    samples = []
    valid, invalid = 0, 0
    for i in row_idx:
        start_dt = df.index[i]
        end_dt = start_dt + timedelta(days=cfg.sample_hours / 24)

        sample = df[(df.index >= start_dt) & (df.index < end_dt)]
        sample = sample.dropna()
        if len(sample) == cfg.sample_hours:
            samples.append(sample)
            valid += 1
        else:
            invalid += 1

    log.info(f"invalid: {invalid}, valid: {valid}")
    for i, sample in enumerate(samples):
        # if i % 100 == 0: print(i)
        if i < len(samples) * (1 - cfg.valid_ratio):
            sample.to_csv(base_path / "data" / "DWD" / "train" / f"{i:05d}.csv")
        else:
            sample.to_csv(base_path / "data" / "DWD" / "test" / f"{i:05d}.csv")


if __name__ == "__main__":
    main()
