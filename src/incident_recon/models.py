"""Facts about a detector file that callers need before running it."""

from __future__ import annotations

from pathlib import Path


def batch_one_only(path) -> bool:
    """True for models that take exactly one image per call.

    CoreML: Ultralytics' CoreML runtime silently uses only the FIRST image of a batch.
    Fixed-shape ONNX (export dynamic=False): the input is fixed at batch 1.
    """
    path = str(Path(path).expanduser())
    if path.endswith(".mlpackage") or path.endswith(".mlmodel"):
        return True
    if path.endswith(".onnx"):
        import onnx
        dim = onnx.load(path, load_external_data=False).graph.input[0].type.tensor_type.shape.dim[0]
        return dim.HasField("dim_value")      # a fixed number, not a named dynamic dimension
    return False
