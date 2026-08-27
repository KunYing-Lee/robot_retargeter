from __future__ import annotations

import mujoco
import numpy as np

from scripts.robot_retarget import RobotRetarget, compute_contact_grounding_offsets


def test_box_ground_height_matches_explicit_corners() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root" pos="0 0 0.5" euler="0.2 -0.4 0.1">
              <freejoint/>
              <geom name="sole" type="box" pos="0.1 0 -0.2"
                    size="0.2 0.1 0.03"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = model.geom("sole").id
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    corners = (
        data.geom_xpos[geom_id]
        + (signs * model.geom_size[geom_id]) @ rotation.T
    )

    retarget = RobotRetarget.__new__(RobotRetarget)
    retarget.model = model
    retarget.configuration = type("Configuration", (), {"data": data})()
    retarget.ground_contact_body_ids = []
    retarget.ground_contact_geom_ids = [geom_id]

    assert np.isclose(
        retarget.minimum_ground_contact_height(),
        np.min(corners[:, 2]),
        atol=1.0e-12,
    )


def test_contact_grounding_closes_support_gap_and_preserves_floor() -> None:
    bottoms = np.asarray(
        [
            (0.10, 0.12),
            (0.08, 0.11),
            (0.03, 0.07),
            (-0.01, 0.02),
        ]
    )
    contacts = np.asarray(
        [
            (True, False),
            (False, False),
            (False, True),
            (False, False),
        ]
    )

    offsets = compute_contact_grounding_offsets(bottoms, contacts)
    registered = bottoms + offsets[:, None]

    np.testing.assert_allclose(registered[0, 0], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.min(registered[2]), 0.0, atol=1.0e-12)
    assert np.min(registered) >= -1.0e-12
    np.testing.assert_allclose(offsets[1], -0.065, atol=1.0e-12)
