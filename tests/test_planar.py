import numpy as np
import torch

from gaussifier_sampler.planar import PLY_PROPERTIES, fit_image, save_ply, to_file_frame


def test_fit_image_takes_the_centre_square_and_resizes() -> None:
    image = torch.rand(3, 300, 500)
    out = fit_image(image, 128)
    assert out.shape == (3, 128, 128)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    square = torch.rand(3, 64, 96)
    assert torch.equal(fit_image(square, 64), square[:, :, 16:80])


def test_save_ply_writes_the_3dgs_vertex_layout(tmp_path) -> None:
    n = 5
    splats = {
        "means": torch.randn(n, 3),
        "log_scales": torch.randn(n, 3),
        "quats": torch.randn(n, 4),
        "opacity_logits": torch.randn(n, 1),
        "sh0": torch.randn(n, 1, 3),
    }
    path = tmp_path / "scene.ply"
    save_ply(splats, path)
    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert f"element vertex {n}".encode() in header
    assert header.count(b"property float") == len(PLY_PROPERTIES)
    rows = np.frombuffer(body, dtype="<f4").reshape(n, len(PLY_PROPERTIES))
    filed = to_file_frame(splats)
    assert np.allclose(rows[:, 0:3], filed["means"].numpy())
    assert np.allclose(rows[:, 3:6], splats["sh0"].reshape(n, 3).numpy())
    assert np.allclose(rows[:, 10:14], filed["quats"].numpy())


def _matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = (q / q.norm()).tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_file_frame_is_the_model_frame_turned_about_y() -> None:
    # After a viewer's load-time 180-degree turn about z, the file frame is the model frame
    # turned 180 degrees about x: upright (y up) and facing +z.
    n = 16
    splats = {"means": torch.randn(n, 3), "quats": torch.randn(n, 4)}
    filed = to_file_frame(splats)
    ry = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    assert torch.allclose(filed["means"], splats["means"] @ ry.T)
    for q, q2 in zip(splats["quats"], filed["quats"], strict=True):
        assert torch.allclose(_matrix(q2), ry @ _matrix(q), atol=1e-5)
