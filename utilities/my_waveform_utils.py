import numpy as np
from simpeg.electromagnetics.time_domain.sources import BaseWaveform
from simpeg.utils import validate_float


class TriangularUTEMWaveform(BaseWaveform):
    """Well-regulated triangular UTEM waveform.

    Parameters
    ----------
    operating_frequency : float
        The operating frequency in Hz.
    """

    def __init__(self, operating_frequency, off_time=np.inf, **kwargs):
        super().__init__(off_time=off_time, has_initial_fields=False, **kwargs)
        self.operating_frequency = operating_frequency

    @property
    def operating_frequency(self):
        """UTEM operating frequency.

        Returns
        -------
        float
            The UTEM operating frequency in Hz.
        """
        return self._operating_frequency

    @operating_frequency.setter
    def operating_frequency(self, value):
        self._operating_frequency = validate_float("operating_frequency", value, min_val=0.)

    def eval(self, time):  # noqa: A003
        
        period = 1 / self.operating_frequency
        slope = 4 * self.operating_frequency

        time = time % period

        if time // (period / 2) == 0:
            return -1. + slope * time
        else:
            return 3 - slope * time


