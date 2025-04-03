from pathlib import Path
from typing import Any

import ILAMB.ilamblib as il
import numpy as np
from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB.Variable import Variable
from netCDF4 import Dataset


def _to_dataset(path: Path, variables: list[Path], attributes: dict[str, Any]) -> None:
    # Dump model intermediate outputs to a file
    with Dataset(path, mode="w") as results:
        results.setncatts(attributes | {"complete": 0})
        for var in variables:
            var.toNetCDF4(results, group="MeanState")
        results.setncattr(
            "complete", 1
        )  # if we get here without failure, flag this as complete


class ConfAMOC(Confrontation):
    def __init__(self, **keywords):
        # Ugly, but this is how we call the Confrontation constructor
        super(ConfAMOC, self).__init__(**keywords)

        # Now we overwrite some things which are different here
        self.regions = ["global"]
        self.layout.regions = self.regions

    def stageData(self, m: ModelResult) -> tuple[Variable, Variable]:
        obs = Variable(
            filename=self.source,
            variable_name=self.variable,
            alternate_vars=self.alternate_vars,
            t0=None if len(self.study_limits) != 2 else self.study_limits[0],
            tf=None if len(self.study_limits) != 2 else self.study_limits[1],
        )
        mod = m.extractTimeSeries(
            self.variable,
            alt_vars=self.alternate_vars,
            initial_time=obs.time[0],
            final_time=obs.time[-1],
        )
        obs, mod = il.MakeComparable(obs, mod, clip_ref=True)
        return obs, mod

    def confront(self, m: ModelResult) -> None:
        obs, mod = self.stageData(m)

        # Compute the mean values over the time period
        obs_mean = obs.integrateInTime(mean=True)
        mod_mean = mod.integrateInTime(mean=True)

        # Rename these for better viewing in the output, appending `global` is
        # something ILAMB uses internally and won't appear in the final output.
        obs_mean.name = "AMOC at 26N global"
        mod_mean.name = "AMOC at 26N global"

        # A bias score based on the relative error in AMOC mean.
        bias_score = Variable(
            name="Bias Score global",
            unit="1",
            data=np.exp(-np.abs(mod_mean.data - obs_mean.data) / obs_mean.data),
        )

        # A Taylor score combining aspects of variability and correlation
        with np.errstate(all="ignore"):
            std0 = obs.data.std()
            std = mod.data.std()
        R0 = 1.0
        R = obs.correlation(mod, ctype="temporal")
        std /= std0
        taylor_score = Variable(
            name="Temporal Distribution Score global",
            unit="1",
            data=4.0 * (1.0 + R.data) / ((std + 1.0 / std) ** 2 * (1.0 + R0)),
        )

        # Write out model intermediate outputs
        _to_dataset(
            Path(self.output_path) / f"{self.name}_{m.name}.nc",
            [mod, mod_mean, bias_score, taylor_score],
            {"name": m.name, "color": m.color, "weight": self.cweight},
        )

        # If this process is the 'master' also write out Benchmark data
        if self.master:
            _to_dataset(
                Path(self.output_path) / f"{self.name}_Benchmark.nc",
                [obs, obs_mean],
                {"name": "Benchmark", "color": "k", "weight": self.cweight},
            )


if __name__ == "__main__":
    c = ConfAMOC(
        name="RAPID",
        source="data/amoc_mon_RAPID_BE_NA_200404-202302.nc",
        variable="amoc",
        alternate_vars=["AMOC"],
        output_path="_tmp",
    )
    m = ModelResult("./FESOM2-REcoM", modelname="FESOM2-REcoM")
    c.confront(m)
