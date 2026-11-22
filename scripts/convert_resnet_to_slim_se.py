"""Warm-start a chess SlimSE_ResNet from a plain ResNet checkpoint on CPU."""

import argparse
import json
from pathlib import Path
import re
import shlex
import sys
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alphazero.games.Chess import ChessGame
from alphazero.models.CNNs import ResNet, SlimSE_ResNet
from alphazero.models.model import create_model


@torch.no_grad()
def transfer(source: ResNet, target: SlimSE_ResNet) -> None:
    target.startBlock.load_state_dict(source.startBlock.state_dict())
    target.valueHead.load_state_dict(source.valueHead.state_dict())
    for old, new in zip(source.backBone, target.backBone, strict=True):
        for name in ("conv1", "bn1", "conv2", "bn2"):
            getattr(new, name).load_state_dict(getattr(old, name).state_dict())

        # sigmoid(0) halves the residual. Double its affine output before gating.
        bn2 = cast(nn.BatchNorm2d, new.bn2)
        bn2.weight.mul_(2)
        bn2.bias.mul_(2)

        exc = cast(nn.Sequential, new.excitation)
        exc_last = cast(nn.Linear, exc[-2])
        exc_last.weight.zero_()
        exc_last.bias.zero_()

    policy_fc = cast(nn.Linear, target.policyHead[-2])
    policy_fc.weight.zero_()
    policy_fc.bias.zero_()


@torch.inference_mode()
def verify(source: ResNet, target: SlimSE_ResNet) -> None:
    game = ChessGame()
    states = [game.encode_for_model()]
    masks = [game.get_legal_actions()]
    for move in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        game.make_move(game.parse_move_str(move))
        states.append(game.encode_for_model())
        masks.append(game.get_legal_actions())

    x = torch.from_numpy(np.stack(states)).float()
    legal = torch.from_numpy(np.stack(masks))
    old_features = source.startBlock(x)
    new_features = target.startBlock(x)
    for old, new in zip(source.backBone, target.backBone, strict=True):
        old_features = old(old_features)
        new_features = new(new_features)
        torch.testing.assert_close(new_features, old_features)

    logits, values = target(x)
    torch.testing.assert_close(values, source(x)[1])
    torch.testing.assert_close(logits, torch.zeros_like(logits), rtol=0, atol=0)
    probabilities = logits.masked_fill(~legal, -torch.inf).softmax(dim=1)
    torch.testing.assert_close(probabilities, legal.float() / legal.sum(dim=1, keepdim=True))


def convert(checkpoint: Path, output: Path, seed: int) -> None:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not re.fullmatch(r"model_\d+\.pt", checkpoint.name):
        raise ValueError("Expected a checkpoint named model_NNN.pt")
    config = json.loads(checkpoint.with_name("args.json").read_text())
    if config.get("architecture", "resnet") != "resnet" or config.get("game_type_str", "chess") != "chess":
        raise ValueError("Expected a plain chess ResNet checkpoint")
    dims = (config["game_input_channels"], config["game_spatial_size"], config["game_action_size"])
    if dims != (ChessGame.input_channels, ChessGame.row_count * ChessGame.col_count, ChessGame.action_size):
        raise ValueError(f"Incompatible chess dimensions: {dims}")

    source = create_model(**config).eval()
    source.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    metadata = {key: config[key] for key in (
        "game_input_channels", "game_spatial_size", "game_action_size",
        "num_resBlocks", "num_channels", "head_hidden_size",
    )}
    metadata.update(game_type_str="chess", architecture="slim_se_resnet",
                    learning_rate=0.001, weight_decay=0.0001,
                    initialization_seed=seed, converted_from=str(checkpoint.resolve()))
    torch.manual_seed(seed)
    target = create_model(**metadata).eval()
    assert isinstance(source, ResNet) and isinstance(target, SlimSE_ResNet)
    transfer(source, target)
    verify(source, target)
    optimizer = torch.optim.AdamW(target.parameters(), lr=metadata["learning_rate"],
                                  weight_decay=metadata["weight_decay"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        stage = Path(staging)
        torch.save(target.state_dict(), stage / checkpoint.name)
        torch.save(optimizer.state_dict(), stage / checkpoint.name.replace("model_", "optimizer_", 1))
        (stage / "args.json").write_text(json.dumps(metadata, indent=4) + "\n")
        output.mkdir()
        for path in stage.iterdir():
            path.rename(output / path.name)

    print(f"Verified and saved {output / checkpoint.name}")
    print(f"python -m alphazero.selfplay chess --resume {shlex.quote(str(output / checkpoint.name))} --learning-rate 0.001")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(args.checkpoint.expanduser(), args.output.expanduser(), args.seed)
