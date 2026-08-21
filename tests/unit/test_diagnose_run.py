"""Resume past step 20 must not abort on the first logged step."""

from minakanushi.training.trainer import TrainLog, _diagnose_run


def _log(step: int, loss: float = 78.5) -> TrainLog:
    return TrainLog(step=step, loss=loss, terms={"state": 1.0}, grad_norm=1.0, traj_error=1.0)


def test_resume_step65_single_log_is_not_constant_loss() -> None:
    assert _diagnose_run([_log(65)]) is None


def test_twenty_identical_losses_still_abort() -> None:
    logs = [_log(step, 1.0) for step in range(1, 21)]
    assert _diagnose_run(logs) == "loss stays constant"
