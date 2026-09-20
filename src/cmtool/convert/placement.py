"""Where each flexure sits on its links: attachment points and axes.

This is pure geometry -- no CAD kernel, no solver -- because three different
consumers need it and only one of them builds solids:

* :mod:`cmtool.cad.mechanism` extrudes strips between these points,
* :mod:`cmtool.solvers.beam_fea` meshes beam elements between them,
* :mod:`cmtool.viz` draws them.

**Pivot matching means the model's characteristic pivot lands on the original
rigid joint** -- not the flexure's midpoint. Those are the same point only for
the small-length model, whose pivot happens to be at the centre. A long segment
pivots at ``(1 - gamma) L`` from its root, so the strip sits asymmetrically about
the joint and most of it lies on the far side. Centring a long segment would put
its pivot roughly ``0.24 L`` from where the kinematics assume it is
(``docs/physics.md`` section 6).
"""

from __future__ import annotations

import numpy as np

from cmtool.convert.base import CompliantMechanism
from cmtool.core.graph import Linkage
from cmtool.core.units import FloatArray


def unit_vector(vector: FloatArray) -> FloatArray:
    """Return ``vector`` scaled to unit length.

    Raises
    ------
    ValueError
        If the vector has zero length, which means two joints coincide and the
        link direction is undefined rather than merely small.
    """
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("cannot normalise a zero-length vector")
    return np.asarray(vector, dtype=float) / norm


def normal_vector(direction: FloatArray) -> FloatArray:
    """Return the in-plane left normal of ``direction`` (rotated +90 degrees)."""
    return np.array([-direction[1], direction[0]], dtype=float)


def attachment_points(
    mechanism: CompliantMechanism, linkage: Linkage | None = None
) -> tuple[dict[str, dict[str, FloatArray]], dict[str, FloatArray]]:
    """Compute per-body flexure attachment points and each flexure's axis.

    Parameters
    ----------
    mechanism
        The converted mechanism, whose ``sizing`` gives each flexure its length,
        host body and pivot fraction.
    linkage
        The rigid linkage the flexures were sized against. Defaults to
        ``mechanism.base``, which is what every caller wants; it is a parameter
        only so a caller that has already resolved it need not look it up twice.

    Returns
    -------
    tuple
        ``(attachments, axes)``. ``attachments[body][joint]`` is the point where
        that body's rigid material meets the flexure at ``joint``, so each
        flexure spans ``attachments[host][joint] -> attachments[other][joint]``.
        ``axes[joint]`` is the unit vector from the joint along its host link,
        pointing away from the joint.
    """
    linkage = linkage if linkage is not None else mechanism.base
    attachments: dict[str, dict[str, FloatArray]] = {b: {} for b in linkage.bodies}
    axes: dict[str, FloatArray] = {}

    for joint_name, joint in linkage.joints.items():
        sized = mechanism.sizing[joint_name]
        host = sized.host_body
        pivot = joint.position_mm

        far = [j for j in linkage.joints_of(host) if j != joint_name]
        toward = linkage.joints[far[0]].position_mm - pivot if far else np.array([1.0, 0.0])
        axis = unit_vector(toward)
        axes[joint_name] = axis

        length = sized.geometry.length_mm
        from_root = sized.pivot_from_root_mm
        other = joint.other(host)
        attachments[host][joint_name] = pivot + axis * from_root
        attachments[other][joint_name] = pivot - axis * (length - from_root)

    return attachments, axes
