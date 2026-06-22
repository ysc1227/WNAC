import math
from torch.optim.lr_scheduler import LRScheduler

class WarmupLin0LrWdScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        max_it: int,
        peak_lr: float = 0.0001,
        wp_ratio: float = 0.03,
        wd: float = 0.05,
        wd_end: float = 0,
        wp0: float = 0.005,
        wpe: float = 0.01,
        last_epoch: int = -1,
    ):
        """
        optimizer: torch optimizer
        max_it   : 총 iteration 수 (max_it-1까지라고 생각하면 됨)
        wp_ratio    : warmup iteration 비율
        peak_lr  : 스케줄의 최대 learning rate
        wd       : 초반 weight decay
        wd_end   : 마지막 weight decay
        wp0      : warmup 시작 비율 (원래 코드의 wp0)
        wpe      : tail 구간에서 내려가는 최종 비율 (원래 코드의 wpe)
        """
        self.max_it = int(max_it)
        self.wp_it = int(max_it * wp_ratio)
        self.peak_lr = float(peak_lr)
        self.wd = float(wd)
        self.wd_end = float(wd_end)
        self.wp0 = float(wp0)
        self.wpe = float(wpe)

        super().__init__(optimizer, last_epoch=last_epoch)

    # lin0 부분만 함수로 빼둔 거
    def _lin0_factor(self, cur_it: int) -> float:
        """
        원래 lr_wd_annealing에서 sche_type == 'lin0' 일 때의
        normalized LR (peak_lr로 곱하기 전)만 구현
        """
        # warmup
        if cur_it < self.wp_it:
            # wp0 -> 1 로 선형 증가
            if self.wp_it <= 0:
                return 1.0
            return self.wp0 + (1.0 - self.wp0) * cur_it / self.wp_it

        # warmup 이후
        denom = max(self.max_it - 1 - self.wp_it, 1)
        pasd = (cur_it - self.wp_it) / denom  # [0, 1]
        pasd = min(max(pasd, 0.0), 1.0)
        rest = 1.0 - pasd

        T = 0.05
        max_rest = 1.0 - T
        if pasd < T:
            # plateau: 1 유지
            return 1.0
        else:
            # 1 -> wpe 로 선형 디케이
            return self.wpe + (1.0 - self.wpe) * rest / max_rest

    def _wd_value(self, cur_it: int) -> float:
        """
        원래 코드의 weight decay cosine annealing 그대로
        """
        denom = max(self.max_it - 1, 1)
        pasd = cur_it / denom  # [0, 1]
        pasd = min(max(pasd, 0.0), 1.0)
        # cosine: wd -> wd_end
        return self.wd_end + (self.wd - self.wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))

    def get_lr(self):
        """
        _LRScheduler가 각 param_group에 적용할 lr 리스트를 반환.
        여기서 optimizer.param_groups를 직접 보면서 lr_sc까지 같이 처리.
        """
        cur_it = self.last_epoch  # step()이 호출될 때마다 0,1,2,... 로 증가

        factor = self._lin0_factor(cur_it)  # [~wp0, 1] + tail 부분
        cur_lr = self.peak_lr * factor

        lrs = []
        for group in self.optimizer.param_groups:
            lr_sc = group.get("lr_sc", 1.0)
            lrs.append(cur_lr * lr_sc)
        return lrs

    def step(self, epoch=None):
        """
        lr는 _LRScheduler의 기본 step 로직을 사용하고,
        weight_decay는 여기에서 따로 업데이트.
        """
        super().step(epoch)

        # weight decay 업데이트
        cur_it = self.last_epoch
        cur_wd = self._wd_value(cur_it)
        for group in self.optimizer.param_groups:
            wd_sc = group.get("wd_sc", 1.0)
            group["weight_decay"] = cur_wd * wd_sc
