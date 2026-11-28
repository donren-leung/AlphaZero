import pytest
import torch

from alphazero.AlphaZero_main import AlphaZero
from alphazero.utils import parse_args


def test_resume_in_checkpoint_folder(tmp_path):
    directory = tmp_path / "chess_r200_p2_gw4_gpw16_b32_e2_resume_parent_model_010"
    directory.mkdir()
    checkpoint = directory / "model_018.pt"
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    torch.save(model.state_dict(), checkpoint)
    torch.save(optimizer.state_dict(), directory / "optimizer_018.pt")
    metadata = directory / "args.json"
    metadata.write_text('{"random_seed": 9, "inference_max_batch_size": 128}')
    args = parse_args(["chess", "--resume", str(checkpoint), "--iter", "2"])
    trainer = AlphaZero(args, model, optimizer, {"architecture": "resnet"})
    assert trainer.artifact_dir == directory
    assert trainer.start_iteration == 19
    assert (args.rollouts, args.processes, args.game_workers, args.games_per_worker,
            args.batch_size, args.epochs, args.seed, args.inference_max_batch_size) == (200, 2, 4, 16, 32, 2, 9, 128)
    assert metadata.read_text() == '{"random_seed": 9, "inference_max_batch_size": 128}'

    (directory / "optimizer_020.pt").touch()
    with pytest.raises(FileExistsError, match="optimizer_020"):
        AlphaZero(args, model, optimizer, {"architecture": "resnet"})


def test_explicit_training_setting_uses_separate_run():
    args = parse_args(["chess", "-i", "2", "--resume", "model_018.pt", "-r", "400"])
    assert not args.resume_in_place
