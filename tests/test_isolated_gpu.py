"""Sampling and benchmark-contract tests; real CUDA smoke test skips without CUDA."""
# mypy: ignore-errors

import importlib.util
import io
import json
from pathlib import Path
import random

import bulletchess
import chess
import chess.pgn
import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "isolated_gpu.py"
spec = importlib.util.spec_from_file_location("isolated_gpu", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

PGN = '''[Event "History"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *

'''
CONFIG = {"game_input_channels": 119, "game_spatial_size": 64, "game_action_size": 4672,
          "num_resBlocks": 1, "num_channels": 8, "head_hidden_size": 1}


def write_pgn(tmp_path, text=PGN * 4):
    path = tmp_path / "games.pgn"
    path.write_text(text)
    return path


def test_sampling_replays_history_and_is_deterministic(tmp_path):
    path = write_pgn(tmp_path)
    first, stats = bench.load_requests(path, 4, 42)
    second, second_stats = bench.load_requests(path, 4, 42)
    assert stats == second_stats
    assert stats["games_read"] == 4
    assert len({r.thread_idx for r in first}) == 4
    game = chess.pgn.read_game(io.StringIO(PGN))
    by_id = {row["request_id"]: row for row in stats["samples"]}
    for request, repeated in zip(first, second):
        assert request.thread_idx == repeated.thread_idx
        np.testing.assert_array_equal(request.state, repeated.state)
        ply = by_id[request.thread_idx]["ply"]
        board = game.board()
        for move in list(game.mainline_moves())[:ply]:
            board.push(move)
        assert len(board.move_stack) == by_id[request.thread_idx]["history_plies"] == ply > 0
        native = bulletchess.Board.from_fen(game.board().fen())
        for move in board.move_stack:
            native.apply(bulletchess.Move.from_uci(move.uci()))
        state = bench.ChessState(native)
        assert not state.get_value_and_terminated()[1]
        assert request.curr_player == (1 if board.turn else -1)
        np.testing.assert_array_equal(request.state, state.encode_for_model())
        np.testing.assert_array_equal(request.legal_mask, state.get_legal_actions())
        # Reconstructing just the final FEN would lose these history planes.
        assert request.state[14:26].any()


def test_terminal_position_excluded_and_empty_game_has_no_sample():
    game = chess.pgn.read_game(io.StringIO("1. f3 e5 2. g4 Qh4# 0-1"))
    for seed in range(20):
        state = bench.sample_board(game, random.Random(seed))
        assert 1 <= bench.board_ply(state.board) <= 3
        assert not state.get_value_and_terminated()[1]
    assert bench.sample_board(chess.pgn.Game(), random.Random(0)) is None


def test_skips_malformed_and_empty_games_and_reports_eof(tmp_path):
    malformed = '[Event "Broken"]\n\n1. e4 e5 2. Bh6 *\n\n'
    empty = '[Event "Empty"]\n\n*\n\n'
    path = write_pgn(tmp_path, malformed + empty + PGN)
    requests, stats = bench.load_requests(path, 1, 42)
    assert len(requests) == 1
    assert stats["malformed"] == 1
    assert stats["no_eligible_position"] == 1
    with pytest.raises(ValueError, match="PGN exhausted"):
        bench.load_requests(path, 2, 42)


def test_batches_share_padded_sequence(tmp_path):
    requests, _ = bench.load_requests(write_pgn(tmp_path), 3, 42)
    batches, total = bench.prepare_batches(requests, [2, 4])
    assert total == 4
    expected = requests + requests[:1]
    for size, group in batches.items():
        assert all(len(batch.thread_idxs) == size for batch in group)
        np.testing.assert_array_equal(np.concatenate([b.thread_idxs for b in group]),
                                      [r.thread_idx for r in expected])
        for batch in group:
            for i, request_id in enumerate(batch.thread_idxs):
                request = next(r for r in requests if r.thread_idx == request_id)
                np.testing.assert_array_equal(batch.padded_legal_actions[i, :batch.lengths[i]],
                                              np.flatnonzero(request.legal_mask))


def test_default_1000_samples_wrap_to_1024(tmp_path):
    requests, _ = bench.load_requests(write_pgn(tmp_path), 1, 42)
    batches, total = bench.prepare_batches(requests * 1000, [64, 128, 256])
    assert total == 1024
    assert [len(batches[size]) for size in [64, 128, 256]] == [16, 8, 4]


def test_checkpoint_configuration_errors(tmp_path):
    checkpoint = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="Checkpoint does not exist"):
        bench.read_model_args(checkpoint, None)
    checkpoint.touch()
    with pytest.raises(ValueError, match="Model configuration does not exist"):
        bench.read_model_args(checkpoint, None)
    config_path = tmp_path / "args.json"
    config_path.write_text(json.dumps(CONFIG | {"resume_from": "ignored"}))
    config, path = bench.read_model_args(checkpoint, None)
    assert config == CONFIG and path == config_path
    config_path.write_text(json.dumps(CONFIG | {"game_input_channels": 3}))
    with pytest.raises(ValueError, match="chess inputs"):
        bench.read_model_args(checkpoint, None)


def test_unavailable_cuda_fails_before_checkpoint_load(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()  # Must never attempt to deserialize it when CUDA is absent.
    (tmp_path / "args.json").write_text(json.dumps(CONFIG))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bench.main(["--checkpoint", str(checkpoint), "--pgn", str(write_pgn(tmp_path))]) == 1
    assert "CUDA is unavailable" in capsys.readouterr().err


def test_production_compute_and_policy_validation_on_cpu(tmp_path):
    requests, _ = bench.load_requests(write_pgn(tmp_path), 2, 42)
    batches, _ = bench.prepare_batches(requests, [2])
    evaluator = bench.DirectEvaluator(bench.create_model(**CONFIG).eval())
    evaluator.device = torch.device("cpu")
    with torch.inference_mode():
        result = evaluator.gpu_compute(batches[2][0])
    bench.validate_policy(batches[2][0], result)
    result.probs[0] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        bench.validate_policy(batches[2][0], result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_smoke(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(bench.create_model(**CONFIG).state_dict(), checkpoint)
    (tmp_path / "args.json").write_text(json.dumps(CONFIG))
    output = tmp_path / "results.json"
    assert bench.main(["--checkpoint", str(checkpoint), "--pgn", str(write_pgn(tmp_path)),
                       "--games", "3", "--batch-sizes", "2", "4", "--warmup", "1",
                       "--iterations", "2", "--repeats", "2", "--selfplay-nps", "10000",
                       "--output-json", str(output)]) == 0
    report = json.loads(output.read_text())
    assert len(report["results"]) == 12
    for row in report["results"]:
        if row["status"] == "unsupported":
            assert row["dtype"] == "bf16"
            continue
        assert row["status"] == "ok"
        assert row["nodes_per_second"] > 0
        assert row["selfplay_ratio"] == row["nodes_per_second"] / 10000
        assert len(row["repetitions"]) == 2
