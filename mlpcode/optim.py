from enum import Enum


class LRSchedulerStrat(Enum):
    na = "No Decay"
    time = "time-based"
    exp = "exponential"
    drop = "drop-based"


class LRScheduler:
    def __init__(
        self,
        alpha=0.1,
        decay_rate=None,
        strategy: LRSchedulerStrat = LRSchedulerStrat.na,
        drop_every_n=10,
    ):
        if decay_rate is None:
            self.decayRate = self.get_default_decay_rate(strategy)
        else:
            if strategy == LRSchedulerStrat.exp and decay_rate > 1.0:
                raise ValueError(
                    "The decay rate must be less than one for exponential decay"
                )
            if strategy == LRSchedulerStrat.drop and decay_rate < 1:
                raise ValueError(
                    "The decay rate must be above 1 for drop based decay (otherwise LR will increase)"
                )
            self.decayRate = decay_rate

        assert decay_rate >= 0.0
        self._alpha0 = alpha
        self.strategy = strategy
        self.stepCount = 0
        self._alpha = alpha
        self.dropEveryN = drop_every_n
        self.step()

    @property
    def value(self):
        return self._alpha

    @staticmethod
    def get_default_decay_rate(strategy: LRSchedulerStrat):
        if strategy == LRSchedulerStrat.na:
            return 0.0
        elif strategy == LRSchedulerStrat.time:
            return 0.0
        elif strategy == LRSchedulerStrat.exp:
            return 1.0
        elif strategy == LRSchedulerStrat.drop:
            return 1.0
        else:
            raise ValueError("Invalid value for LR Decay Strategy")

    def step(self):
        if self.strategy == LRSchedulerStrat.na:
            self._alpha = self._alpha0
        elif self.strategy == LRSchedulerStrat.time:
            self._alpha = 1.0 / (1 + self.decayRate * self.stepCount) * self._alpha0
        elif self.strategy == LRSchedulerStrat.exp:
            self._alpha = self._alpha0 * (self.decayRate ** self.stepCount)
        elif self.strategy == LRSchedulerStrat.drop:
            if (self.stepCount + 1) % self.dropEveryN == 0:
                self._alpha = self._alpha0 / self.decayRate
        else:
            raise ValueError()
        self.stepCount += 1
