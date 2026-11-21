import json
import sys

import pytest
import torch

from alphazero import AlphaZero_main as training
from alphazero.batching.Pool import GPU_AZ_Worker
from alphazero.models.model import create_model
from alphazero.models.CNNs import ResNet, SE_ResNet


@pytest.mark.parametrize("architecture", ["resnet", "se_resnet"])
def test_training_and_worker_checkpoint_round_trip(architecture, tmp_path, monkeypatch):
    # Keep run artifacts local and exercise the normal training entry point on CPU.
    monkeypatch.setattr(training, "__file__", str(tmp_path / "src/alphazero/AlphaZero_main.py"))
    monkeypatch.setattr(sys, "argv", [
        "selfplay", "ttt", "--blocks", "2", "--channels", "8", "--head-channels", "4",
        *(["--architecture", architecture] if architecture == "se_resnet" else []),
    ])
    runs = []
    monkeypatch.setattr(training.AlphaZero, "learn", lambda self: runs.append(self))
    training.main()
    run = runs.pop()
    folder = run.artifact_dir
    metadata = json.loads((folder / "args.json").read_text())
    assert metadata["architecture"] == architecture
    assert "state_dict" not in metadata
    assert folder.name.endswith("_se_resnet") == (architecture == "se_resnet")
    assert type(run.model) is (SE_ResNet if architecture == "se_resnet" else ResNet)

    boards = torch.randn(3, 3, 3, 3)
    policy, value = run.model(boards)
    assert policy.shape == (3, 9)
    assert value.shape == (3, 1)
    (policy.sum() + value.sum()).backward()
    assert all(parameter.grad is not None for parameter in run.model.parameters())
    run.optimizer.step()

    checkpoint = folder / "model_003.pt"
    torch.save(run.model.state_dict(), checkpoint)
    torch.save(run.optimizer.state_dict(), folder / "optimizer_003.pt")
    if architecture == "resnet":
        del metadata["architecture"]  # Pre-factory args.json had no architecture field, so we simulate that here.
        (folder / "args.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(sys, "argv", ["selfplay", "ttt", "--resume", str(checkpoint)])
    training.main()
    resumed = runs.pop()
    assert resumed.start_iteration == 4
    assert resumed.optimizer.state_dict()["state"]

    def evaluate_with_worker(game_type, rollouts, processes, workers, games, model_args, seed):
        worker = GPU_AZ_Worker([], [], None, model_args) # type: ignore 
        return worker.model.eval()(boards)

    monkeypatch.setattr(training, "self_play", evaluate_with_worker)
    result = resumed.self_play(resumed.args.game_type, 1, 1, 1, 1, 42)
    torch.testing.assert_close(result, run.model.eval()(boards))

    sys.argv.extend(["--architecture", "se_resnet" if architecture == "resnet" else "resnet"])
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        training.main()


def test_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown model architecture"):
        create_model(architecture="typo")
