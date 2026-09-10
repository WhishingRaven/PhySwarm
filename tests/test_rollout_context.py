"""actor 관측과 분리된 physics rollout side-channel 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from learning.physics_loss import project_torch_parameters
from learning.rollout_context import (
    PhysicsContextBuffer,
    body_to_world_vectors,
    build_physics_transition,
    build_micro_transition,
    phase_indices_from_one_hot,
    record_physics_transition,
)
from learning.trajectory_loss import (
    exact_micro_loss_from_context,
    exact_physics_loss_from_context,
)
from physics.layouts import get_parameter_layout


def test_micro_transition_builder_copies_and_validates_canonical_shapes() -> None:
    velocity = np.ones((1, 2, 2), dtype=np.float32)
    context = build_micro_transition(
        executed_velocity=velocity,
        field_bases=np.ones((1, 2, 3, 2), dtype=np.float32),
        density=np.full((1, 2), 0.25, dtype=np.float32),
        density_gradient=np.zeros((1, 2, 2), dtype=np.float32),
        active_mask=np.array([[True, False]]),
    )
    velocity.fill(9.0)

    np.testing.assert_array_equal(context["executed_velocity"], 1.0)
    assert set(context) == {
        "executed_velocity",
        "field_bases",
        "density",
        "density_gradient",
        "active_mask",
    }


def test_macro_transition_uses_arena_vectors_and_explicit_state_endpoints() -> None:
    body_bases = np.array(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]],
        dtype=np.float32,
    )
    headings = np.array([[np.pi / 2.0, 0.0]], dtype=np.float32)
    global_bases = body_to_world_vectors(body_bases, headings)
    np.testing.assert_allclose(global_bases[0, 0, 0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(global_bases[0, 0, 1], [-1.0, 0.0], atol=1e-6)

    active = np.array([[True, True]])
    phases = phase_indices_from_one_hot(
        np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32),
        active_mask=active,
    )
    context = build_physics_transition(
        executed_velocity=np.zeros((1, 2, 2), dtype=np.float32),
        field_bases=body_bases,
        density=np.ones((1, 2), dtype=np.float32),
        density_gradient=np.zeros((1, 2, 2), dtype=np.float32),
        active_mask=active,
        field_bases_global=global_bases,
        position_start=np.zeros((1, 2, 2), dtype=np.float32),
        position_end=np.full((1, 2, 2), 0.1, dtype=np.float32),
        phase_start=phases,
        phase_end=phases,
        active_start=active,
        active_end=active,
        reaction_gates=np.zeros((1, 2, 0), dtype=np.float32),
    )
    assert context["phase_start"].dtype == np.int64
    assert context["reaction_gates"].shape == (1, 2, 0)
    assert not np.shares_memory(context["field_bases_global"], global_bases)

    with pytest.raises(ValueError, match="exactly one phase"):
        phase_indices_from_one_hot(
            np.array([[[1, 1, 0]]], dtype=np.float32),
            active_mask=np.array([[True]]),
        )


def test_physics_context_tracks_same_chunks_as_episode_major_ppo_buffer() -> None:
    episode_length, num_envs, num_agents = 4, 2, 3
    episode: dict[str, np.ndarray] = {}
    for step in range(episode_length):
        value = np.zeros((num_envs, num_agents, 2), dtype=np.float32)
        for environment in range(num_envs):
            value[environment, :, 0] = 100 * environment + 10 * step + np.arange(num_agents)
        record_physics_transition(
            episode,
            {"executed_velocity": value},
            step=step,
            episode_length=episode_length,
            num_envs=num_envs,
            num_agents=num_agents,
        )

    buffer = PhysicsContextBuffer(
        episode_length=episode_length,
        buffer_size=2,
        num_agents=num_agents,
    )
    buffer.insert(episode, np.array([0, 1]))
    sampled = buffer.sample_chunks([0, 2, 4, 6], data_chunk_length=2)

    assert sampled is not None
    values = sampled["executed_velocity"]
    assert values.shape == (4, 2, num_agents, 2)
    np.testing.assert_array_equal(values[0, :, 0, 0], [0.0, 10.0])
    np.testing.assert_array_equal(values[2, :, 0, 0], [100.0, 110.0])


def test_physics_context_rejects_shape_drift_and_episode_crossing() -> None:
    episode: dict[str, np.ndarray] = {}
    record_physics_transition(
        episode,
        {"density": np.ones((1, 2), dtype=np.float32)},
        step=0,
        episode_length=3,
        num_envs=1,
        num_agents=2,
    )
    with pytest.raises(ValueError, match="changed shape"):
        record_physics_transition(
            episode,
            {"density": np.ones((1, 2, 1), dtype=np.float32)},
            step=1,
            episode_length=3,
            num_envs=1,
            num_agents=2,
        )

    buffer = PhysicsContextBuffer(episode_length=3, buffer_size=2, num_agents=2)
    buffer.insert(episode, np.array([0]))
    with pytest.raises(ValueError, match="crosses an episode"):
        buffer.sample_chunks([2], data_chunk_length=2)


def test_buffered_micro_loss_realigns_flat_actor_parameters() -> None:
    layout = get_parameter_layout("rescue", contract="legacy")
    batch, time, agents = 2, 3, 2
    raw = torch.zeros(
        (batch * time * agents, layout.action_dim), requires_grad=True
    )
    parameters = project_torch_parameters(raw, layout).as_named_dict(layout)
    context = {
        "executed_velocity": np.zeros((batch, time, agents, 2), dtype=np.float32),
        "field_bases": np.ones(
            (batch, time, agents, len(layout.advection_names), 2),
            dtype=np.float32,
        ),
        "density": np.full((batch, time, agents), 0.2, dtype=np.float32),
        "density_gradient": np.zeros(
            (batch, time, agents, 2), dtype=np.float32
        ),
        "active_mask": np.ones((batch, time, agents), dtype=bool),
    }

    result = exact_micro_loss_from_context(context, parameters, layout)
    assert result.loss.item() > 0.0
    result.loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_joint_context_loss_connects_macro_trajectory_to_actor_parameters() -> None:
    """T+1 KDE, field interpolation, learned/event reaction을 한 graph로 검증한다."""

    layout = get_parameter_layout("foraging", contract="legacy")
    batch, time, agents = 1, 2, 3
    raw = torch.zeros(
        (batch * time * agents, layout.action_dim), requires_grad=True
    )
    parameters = project_torch_parameters(raw, layout).as_named_dict(layout)

    state_0 = np.array([[-0.25, 0.0], [0.0, 0.0], [0.25, 0.0]], dtype=np.float32)
    state_1 = state_0 + np.array([0.01, 0.0], dtype=np.float32)
    state_2 = state_1 + np.array([0.01, 0.0], dtype=np.float32)
    phase_0 = np.array([0, 1, 2], dtype=np.int64)
    # agent 1 picks up and agent 2 drops; these are the learned reaction edges.
    phase_1 = np.array([0, 2, 0], dtype=np.int64)
    phase_2 = phase_1.copy()
    bases = np.zeros((batch, time, agents, 4, 2), dtype=np.float32)
    bases[..., 0, 0] = 1.0
    bases[..., 1, 1] = 1.0
    bases[..., 2, 0] = -1.0
    bases[..., 3, 1] = -1.0
    gates = np.zeros((batch, time, agents, 2), dtype=np.float32)
    gates[0, 0, 1, 0] = 1.0
    gates[0, 0, 2, 1] = 1.0
    context = {
        "executed_velocity": np.zeros((batch, time, agents, 2), dtype=np.float32),
        "field_bases": bases.copy(),
        "field_bases_global": bases.copy(),
        "density": np.full((batch, time, agents), 0.2, dtype=np.float32),
        "density_gradient": np.zeros((batch, time, agents, 2), dtype=np.float32),
        "active_mask": np.ones((batch, time, agents), dtype=bool),
        "position_start": np.stack((state_0, state_1))[None],
        "position_end": np.stack((state_1, state_2))[None],
        "phase_start": np.stack((phase_0, phase_1))[None],
        "phase_end": np.stack((phase_1, phase_2))[None],
        "active_start": np.ones((batch, time, agents), dtype=bool),
        "active_end": np.ones((batch, time, agents), dtype=bool),
        "reaction_gates": gates,
    }

    result = exact_physics_loss_from_context(
        context,
        parameters,
        layout,
        "foraging",
        mode="exact_joint",
        dt=0.1,
        bandwidth=0.2,
        grid_shape=(7, 11),
    )
    assert result.micro_result is not None
    assert result.macro_result is not None
    assert torch.isfinite(result.breakdown.total)
    assert result.breakdown.macro.item() > 0.0
    torch.testing.assert_close(
        result.macro_result.reaction_source.sum(dim=2),
        torch.zeros_like(result.macro_result.reaction_source[:, :, 0]),
        atol=1e-5,
        rtol=0.0,
    )
    result.breakdown.total.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad) > 0


def test_navigation_exact_loss_rebuilds_shape_field_with_beta_gradient() -> None:
    layout = get_parameter_layout("navigation", contract="legacy")
    batch, time, agents = 1, 1, 2
    raw = torch.zeros(
        (batch * time * agents, layout.action_dim), requires_grad=True
    )
    parameters = project_torch_parameters(raw, layout).as_named_dict(layout)
    bases = np.zeros((batch, time, agents, 2, 2), dtype=np.float32)
    context = {
        "executed_velocity": np.zeros((batch, time, agents, 2), dtype=np.float32),
        "field_bases": bases.copy(),
        "field_bases_global": bases.copy(),
        "density": np.full((batch, time, agents), 0.2, dtype=np.float32),
        "density_gradient": np.zeros((batch, time, agents, 2), dtype=np.float32),
        "active_mask": np.ones((batch, time, agents), dtype=bool),
        "heading_start": np.zeros((batch, time, agents), dtype=np.float32),
        "navigation_shape_geometry": np.array(
            [[[[0.1, 0.2, 0.15], [-0.1, -0.2, 0.15]]]], dtype=np.float32
        ),
    }
    result = exact_physics_loss_from_context(
        context,
        parameters,
        layout,
        "navigation",
        mode="exact_micro",
        dt=0.1,
    )
    result.breakdown.total.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    # beta is the final Navigation action channel. A detached rollout basis would
    # make this exactly zero, which is the regression this test prevents.
    assert torch.count_nonzero(raw.grad[:, -1]) > 0


def test_foraging_paper_pick_rate_controls_trail_to_home_reaction() -> None:
    """Table S5의 pickup 한 채널이 S128 trail->home에도 gradient를 준다."""

    layout = get_parameter_layout("foraging", contract="paper")
    raw = torch.zeros((1, layout.action_dim), requires_grad=True)
    parameters = project_torch_parameters(raw, layout).as_named_dict(layout)
    position = np.zeros((1, 1, 1, 2), dtype=np.float32)
    context = {
        "field_bases_global": np.zeros((1, 1, 1, 4, 2), dtype=np.float32),
        "position_start": position.copy(),
        "position_end": position.copy(),
        "phase_start": np.array([[[3]]], dtype=np.int64),
        "phase_end": np.array([[[2]]], dtype=np.int64),
        "active_start": np.ones((1, 1, 1), dtype=bool),
        "active_end": np.ones((1, 1, 1), dtype=bool),
        "reaction_gates": np.array([[[[1.0, 0.0]]]], dtype=np.float32),
    }

    result = exact_physics_loss_from_context(
        context,
        parameters,
        layout,
        "foraging",
        mode="exact_macro",
        phase_contract="paper",
        dt=0.1,
        bandwidth=0.2,
        grid_shape=(7, 11),
    )
    assert result.macro_result is not None
    torch.testing.assert_close(
        result.macro_result.reaction_source.sum(dim=2),
        torch.zeros_like(result.macro_result.reaction_source[:, :, 0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    result.breakdown.total.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert abs(float(raw.grad[0, 5])) > 1.0e-8
