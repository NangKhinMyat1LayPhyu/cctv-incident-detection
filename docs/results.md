# Results

## Detector

UA-DETRAC validation set, as is and with every image degraded like poor CCTV footage
(low resolution, grayscale, darkness, blur, heavy compression):

| Detector | mAP@50, clean | mAP@50, degraded |
|---|---|---|
| YOLO11n, COCO pretrained | 0.538 | 0.398 |
| + fine-tuned on UA-DETRAC (with pseudo-labels) | 0.659 | 0.548 |
| **+ synthetic CCTV degradation** | **0.666** | **0.591** |

- Fine-tuning is the large gain: +0.12 mAP@50 on clean frames, +0.15 on degraded ones.
- Training with degraded copies adds +0.04 on degraded frames, mostly for motorcycles and
  people, and shrinks the drop on bad footage from 0.11 to 0.075.

## Crash detection

Held-out set: 40 accident clips and 30 normal clips from TAD, never used for tuning.

| Detector | Recall | Precision | False alarms / h |
|---|---|---|---|
| v1: sudden-stop + stopped-vehicles rules | 0.50 | 0.87 | 31 |
| **v2: + queue filter** | **0.50** | **0.95** | **10** |

The v2 queue filter removes two of three false alarms (traffic queues at red lights) without
losing a detection. Per-clip results: `results/heldout_v2_results.csv`.

## Live system

The whole chain runs frame by frame as video arrives: 11-12 ms per frame (95% under 17 ms),
against a 33-40 ms budget at 25-30 fps.

Several cameras in one process, frames detected as one batch:

| Cameras | Batched: frames/s | One at a time: frames/s |
|---|---|---|
| 1 | 79 | 85 |
| 2 | 117 | 81 |
| 4 | 122 | 90 |
| 8 | 139 | 99 |

- Batching gives about 1.4x the throughput; real time holds for 2 cameras at 30 fps or 4 at 25 fps.
- Tested on real RTSP streams: a cut stream reconnects within seconds while other cameras keep running.

## Exported models

Each export checked for speed and for crash detection on the full held-out set:

| Model | Runtime | Latency mean / p95 | Recall | Precision |
|---|---|---|---|---|
| PyTorch FP32 | MPS (GPU) | 6.3 / 11.7 ms | 0.57 | 0.85 |
| ONNX FP32 | ONNX Runtime, CPU | 11.2 / 12.3 ms | 0.57 | 0.85 |
| **CoreML FP16** | CoreML | **5.3 / 5.6 ms** | 0.57 | 0.85 |
| CoreML INT8 weights | CoreML | 5.4 / 5.8 ms | 0.65 | 0.87 |

These runs used a different machine from the crash-detection table above, which is why their recall
and precision differ slightly from 0.50 / 0.95 (see the last point).

- **CoreML FP16** for one camera: as fast as the GPU and twice as steady.
- **Batched PyTorch** for several cameras (4.3 ms per frame at batch 4).
- **ONNX** reproduces PyTorch exactly and is 30% faster than PyTorch on CPU.
- The same weights flip 6-11 of 70 borderline clips across runtimes (FP16, INT8, another GPU);
  a study on the dev set traced most flips to detection and tracking rather than the rules.
