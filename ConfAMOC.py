import ILAMB.ilamblib as il
import numpy as np
from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB.Variable import Variable


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


if __name__ == "__main__":
    c = ConfAMOC(
        source="data/amoc_mon_RAPID_BE_NA_200404-202302.nc",
        variable="amoc",
        alternate_vars=["AMOC"],
    )
    m = ModelResult("./FESOM2-REcoM", modelname="FESOM2-REcoM")
    obs, mod = c.stageData(m)
    print(obs)
    print(mod)

    obs_mean = obs.integrateInTime(mean=True)
    mod_mean = obs.integrateInTime(mean=True)

    rel_error = np.abs(mod_mean.data - obs_mean.data) / obs_mean.data
    score = np.exp(-rel_error)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    obs.plot(ax=ax)

    fig.savefig("amoc.png")
