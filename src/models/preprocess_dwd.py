import random
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Collection

import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger as log
from wetterdienst import DWDStationRequest, Parameter, TimeResolution


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


@dataclass
class DWDConfig:
    res: str = "hourly"
    start_date: str = "2005-01-01"
    end_date: str = "2020-06-30"
    stations: Collection[int] = field(default_factory=lambda: [2290, 2542])
    valid_ratio: float = 0.2
    sample_hours: int = 32


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=DWDConfig)


@hydra.main(config_name="config")
def main(cfg: DWDConfig) -> None:
    base_path = Path(hydra.utils.get_original_cwd())
    (base_path / "data" / "DWD" / "test").mkdir(exist_ok=True, parents=True)
    (base_path / "data" / "DWD" / "train").mkdir(exist_ok=True, parents=True)

    # hohenpeissenberg, kaufbeuren
    # target, supplier(s)
    # ids: (2290, 2542)

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
        df2 = df[["ELEMENT", "DATE", "VALUE"]]
        df3 = df2.pivot(index="DATE", columns="ELEMENT", values="VALUE")
        dfs.append(df3[["PRECIPITATION_HEIGHT", "TEMPERATURE_AIR_200"]])

    # merge and keep indices
    log.warning(f"Only using 2 stations for now (target, source): {cfg.stations[:2]}")
    df = dfs[0].merge(dfs[1], how="inner", left_index=True, right_index=True)
    df.columns = ["TARGET_PRCP", "TARGET_TEMP", "SOURCE_PRCP", "SOURCE_TEMP"]

    log.debug(f"Sample:\n{df.head()}")

    n = len(df)
    rows = range(n)
    row_idx = sorted(random.choices(rows, k=n // 5))

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
