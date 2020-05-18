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
        decay_rate: float = None,
        strategy: LRSchedulerStrat = LRSchedulerStrat.na,
        drop_every_n=10,
    ):
        if decay_rate is None:
            self.__decayRate = self.get_default_decay_rate(strategy)
        else:
            if strategy == LRSchedulerStrat.exp and decay_rate > 1.0:
                raise ValueError(
                    "The decay rate must be less than one for exponential decay"
                )
            if strategy == LRSchedulerStrat.drop and decay_rate < 1:
                raise ValueError(
                    "The decay rate must be above 1 for drop based decay (otherwise LR will increase)"
                )
            self.__decayRate = decay_rate

        assert self.__decayRate >= 0.0
        self.__alpha0 = alpha
        self.strategy = strategy
        self.__stepCount = 0
        self.__alpha = alpha
        self.__dropEveryN = drop_every_n
        self.step()

    @property
    def value(self) -> float:
        return self.__alpha

    @staticmethod
    def get_default_decay_rate(strategy: LRSchedulerStrat) -> float:
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

    def step(self) -> None:
        if self.strategy == LRSchedulerStrat.na:
            self.__alpha = self.__alpha0
        elif self.strategy == LRSchedulerStrat.time:
            self.__alpha = (
                1.0 / (1 + self.__decayRate * self.__stepCount) * self.__alpha0
            )
        elif self.strategy == LRSchedulerStrat.exp:
            self.__alpha = self.__alpha0 * (self.__decayRate ** self.__stepCount)
        elif self.strategy == LRSchedulerStrat.drop:
            if (self.__stepCount + 1) % self.__dropEveryN == 0:
                self.__alpha = self.__alpha0 / self.__decayRate
        else:
            raise ValueError()

        self.__stepCount += 1
