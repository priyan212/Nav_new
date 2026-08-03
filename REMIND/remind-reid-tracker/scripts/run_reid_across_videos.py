from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_video_tracking import (
    FrameSource,
    _configure,
    _default_output_dir,
    _open_writer,
    render_frame,
    resolve_yolo_model,
)


@dataclass
class ObjectRecord:
    object_id: int
    class_name: str | None
    first_video: int
    first_frame_id: int
    first_timestamp: float
    seen_in_video1: bool = False
    seen_in_video2: bool = False
    v1_hits: int = 0
    v2_hits: int = 0
    v2_first_frame_id: int | None = None
    v2_first_timestamp: float | None = None


def _class_name_by_det_id(detections) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    for det in detections or []:
        out[int(getattr(det, "detection_id", -1))] = getattr(det, "class_name", None)
    return out


def _process_video(
    *,
    pipeline,
    frame_source: FrameSource,
    max_frames: int | None,
    frame_id_offset: int,
    timestamp_offset: float,
    save_fps: float,
    output_video_path: Path | None,
    frames_csv_path: Path,
    detections_jsonl_path: Path,
    mask_alpha: float,
    video_label: str,
    registry: dict[int, ObjectRecord],
    video_index: int,
    show_viewer: bool,
) -> tuple[int, float, int]:
    """Runs REMIND over one video through the shared pipeline/memory instance.

    Returns (last_frame_id, last_timestamp, processed_count).
    """
    writer: cv2.VideoWriter | None = None
    last_frame_id = frame_id_offset
    last_timestamp = timestamp_offset
    processed = 0

    with open(frames_csv_path, "w", newline="", encoding="utf-8") as csv_fh, open(
        detections_jsonl_path, "w", encoding="utf-8"
    ) as jsonl_fh:
        csv_writer = csv.DictWriter(
            csv_fh,
            fieldnames=[
                "frame_idx", "frame_id", "timestamp", "name",
                "detections", "matches", "created", "ambiguous", "provisional",
                "visible", "reidentified_from_video1", "elapsed_seconds",
            ],
        )
        csv_writer.writeheader()

        if show_viewer:
            cv2.namedWindow(f"REMIND tracking - {video_label}", cv2.WINDOW_NORMAL)

        try:
            for local_idx, item in enumerate(frame_source.iter_frames(max_frames=max_frames)):
                frame_id = frame_id_offset + local_idx
                timestamp = timestamp_offset + item.timestamp
                t0 = perf_counter()
                p_out, _a_out, u_out = pipeline.process_frame(
                    frame=item.frame, frame_id=int(frame_id), timestamp=float(timestamp)
                )
                elapsed = perf_counter() - t0
                processed += 1
                fps_now = 1.0 / elapsed if elapsed > 0 else 0.0

                class_by_det = _class_name_by_det_id(p_out.detections or [])
                reid_hits_this_frame = 0

                for m in getattr(u_out, "matches", []) or []:
                    oid = int(m["object_id"])
                    det_id = int(m["det_id"])
                    rec = registry.get(oid)
                    if rec is None:
                        rec = ObjectRecord(
                            object_id=oid, class_name=class_by_det.get(det_id),
                            first_video=video_index, first_frame_id=frame_id, first_timestamp=timestamp,
                        )
                        registry[oid] = rec
                    if video_index == 1:
                        rec.seen_in_video1 = True
                        rec.v1_hits += 1
                    else:
                        rec.seen_in_video2 = True
                        rec.v2_hits += 1
                        if rec.v2_first_frame_id is None:
                            rec.v2_first_frame_id = frame_id
                            rec.v2_first_timestamp = timestamp
                        if rec.first_video == 1:
                            reid_hits_this_frame += 1

                for c in getattr(u_out, "created", []) or []:
                    oid = int(c["object_id"])
                    det_id = int(c["det_id"])
                    if oid not in registry:
                        registry[oid] = ObjectRecord(
                            object_id=oid, class_name=class_by_det.get(det_id),
                            first_video=video_index, first_frame_id=frame_id, first_timestamp=timestamp,
                        )
                    if video_index == 1:
                        registry[oid].seen_in_video1 = True
                        registry[oid].v1_hits += 1
                    else:
                        registry[oid].seen_in_video2 = True
                        registry[oid].v2_hits += 1
                        if registry[oid].v2_first_frame_id is None:
                            registry[oid].v2_first_frame_id = frame_id
                            registry[oid].v2_first_timestamp = timestamp

                summary = getattr(u_out, "summary", {}) or {}
                render_base = ((getattr(p_out, "debug", {}) or {}).get("frame_aligned_bgr", None))
                if render_base is None:
                    render_base = item.frame
                header = (
                    f"[{video_label}] {item.name} | frame={frame_id} | det={len(p_out.detections or [])} | "
                    f"visible={summary.get('n_visible', 0)} | new={summary.get('n_created', 0)} | "
                    f"reid={reid_hits_this_frame} | amb={summary.get('n_ambiguous', 0)} | {fps_now:.2f} FPS"
                )
                rendered, det_details = render_frame(
                    render_base, p_out.detections or [], u_out, header=header, alpha=float(mask_alpha)
                )

                if output_video_path is not None:
                    if writer is None:
                        writer = _open_writer(output_video_path, rendered.shape, save_fps)
                    writer.write(rendered)

                csv_writer.writerow({
                    "frame_idx": int(item.frame_idx), "frame_id": int(frame_id), "timestamp": float(timestamp),
                    "name": item.name, "detections": int(len(p_out.detections or [])),
                    "matches": int(summary.get("n_matches", 0)), "created": int(summary.get("n_created", 0)),
                    "ambiguous": int(summary.get("n_ambiguous", 0)), "provisional": int(summary.get("n_provisional", 0)),
                    "visible": int(summary.get("n_visible", 0)), "reidentified_from_video1": int(reid_hits_this_frame),
                    "elapsed_seconds": float(elapsed),
                })
                jsonl_fh.write(json.dumps({
                    "frame_idx": int(item.frame_idx), "frame_id": int(frame_id), "timestamp": float(timestamp),
                    "name": item.name, "detections": det_details,
                }, ensure_ascii=True) + "\n")

                if show_viewer:
                    cv2.imshow(f"REMIND tracking - {video_label}", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print(f"[REMIND-REID] Stopped by user during {video_label}.")
                        break

                last_frame_id = frame_id
                last_timestamp = timestamp
        finally:
            if writer is not None:
                writer.release()
            if show_viewer:
                cv2.destroyWindow(f"REMIND tracking - {video_label}")

    return last_frame_id, last_timestamp, processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run REMIND across two videos of the SAME scene through one shared identity "
            "memory: video1 builds up the object catalogue, video2 (a different viewpoint/"
            "revisit) is matched against it, reproducing the paper's leave-and-re-enter "
            "scenario (Fig. 1) and reporting which objects were correctly re-identified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene", help="Name for this run (used for output folder naming only).")
    parser.add_argument("yolo_model", help="YOLO segmentation model file name located inside the yolo/ folder.")
    parser.add_argument("--video1", type=Path, required=True, help="First visit video (builds the object memory).")
    parser.add_argument("--video2", type=Path, required=True, help="Re-entry video (matched against video1's memory).")
    parser.add_argument("--yolo-dir", type=Path, default=REPO_ROOT / "yolo")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default_config.yaml")
    parser.add_argument("--override-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--show-viewer", action="store_true")
    io_group.add_argument("--max-frames", type=int, default=None, help="Cap applied independently to each video.")
    io_group.add_argument("--input-video-fps", type=float, default=None)
    io_group.add_argument("--stride", type=int, default=1)
    io_group.add_argument("--output-fps", type=float, default=30.0)
    io_group.add_argument(
        "--gap-seconds", type=float, default=120.0,
        help="Simulated real-world gap (seconds) between end of video1 and start of video2, for logging/timestamps only.",
    )

    yolo = parser.add_argument_group("YOLO")
    yolo.add_argument("--yolo-conf", type=float, default=0.25)
    yolo.add_argument("--yolo-iou", type=float, default=0.7)
    yolo.add_argument("--yolo-imgsz", type=int, default=960)
    yolo.add_argument("--max-det", type=int, default=100)
    yolo.add_argument("--classes", default=None)
    yolo.add_argument("--mask-erosion-px", type=int, default=0)
    yolo.add_argument("--mask-erosion-iters", type=int, default=1)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    runtime.add_argument("--dino-model-label", default=None)
    runtime.add_argument("--verbose-timing", action="store_true")
    runtime.add_argument("--mask-alpha", type=float, default=0.42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    video1 = args.video1.expanduser().resolve()
    video2 = args.video2.expanduser().resolve()
    if not video1.exists():
        raise SystemExit(f"error: video1 not found: {video1}")
    if not video2.exists():
        raise SystemExit(f"error: video2 not found: {video2}")

    scene_name = str(args.scene).strip()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else _default_output_dir(video1, scene=f"{scene_name}_reid")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        args.yolo_model = resolve_yolo_model(args.yolo_model, models_dir=args.yolo_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    print("[REMIND-REID] Initializing models...")
    from pipeline.initialization import initialize_system
    from pipeline.reid_pipeline import ReIDPipeline

    config = _configure(args, output_dir)
    ctx = initialize_system(config)
    pipeline = ReIDPipeline(ctx)
    print(f"[REMIND-REID] Scene: {scene_name}")
    print(f"[REMIND-REID] Video1 (first visit): {video1}")
    print(f"[REMIND-REID] Video2 (re-entry):     {video2}")
    print(f"[REMIND-REID] Output: {output_dir}")
    print(f"[REMIND-REID] Device: {ctx.device}")

    save_fps = max(0.1, float(args.output_fps))
    registry: dict[int, ObjectRecord] = {}

    fs1 = FrameSource(video1, stride=args.stride, input_video_fps=args.input_video_fps, frames_timestamp_fps=args.output_fps)
    t_run = perf_counter()
    last_frame_id, last_timestamp, n1 = _process_video(
        pipeline=pipeline, frame_source=fs1, max_frames=args.max_frames,
        frame_id_offset=0, timestamp_offset=0.0, save_fps=save_fps,
        output_video_path=output_dir / "video1_tracking.mp4",
        frames_csv_path=output_dir / "video1_frames.csv",
        detections_jsonl_path=output_dir / "video1_detections.jsonl",
        mask_alpha=args.mask_alpha, video_label="VIDEO1-FIRST-VISIT",
        registry=registry, video_index=1, show_viewer=args.show_viewer,
    )
    n_catalogued = sum(1 for r in registry.values() if r.seen_in_video1)
    print(f"[REMIND-REID] Video1 done: {n1} frames processed, {n_catalogued} objects catalogued.")

    fs2 = FrameSource(video2, stride=args.stride, input_video_fps=args.input_video_fps, frames_timestamp_fps=args.output_fps)
    _, _, n2 = _process_video(
        pipeline=pipeline, frame_source=fs2, max_frames=args.max_frames,
        frame_id_offset=last_frame_id + 1, timestamp_offset=last_timestamp + float(args.gap_seconds),
        save_fps=save_fps,
        output_video_path=output_dir / "video2_tracking.mp4",
        frames_csv_path=output_dir / "video2_frames.csv",
        detections_jsonl_path=output_dir / "video2_detections.jsonl",
        mask_alpha=args.mask_alpha, video_label="VIDEO2-RE-ENTRY",
        registry=registry, video_index=2, show_viewer=args.show_viewer,
    )
    total_seconds = perf_counter() - t_run

    v1_ids = {oid for oid, r in registry.items() if r.seen_in_video1}
    v2_ids = {oid for oid, r in registry.items() if r.seen_in_video2}
    reidentified = sorted(v1_ids & v2_ids)
    missed = sorted(v1_ids - v2_ids)
    new_in_v2 = sorted(v2_ids - v1_ids)

    report = {
        "scene": scene_name,
        "video1": str(video1),
        "video2": str(video2),
        "video1_frames_processed": int(n1),
        "video2_frames_processed": int(n2),
        "objects_catalogued_in_video1": len(v1_ids),
        "objects_reidentified_in_video2": len(reidentified),
        "objects_missed_in_video2": len(missed),
        "objects_new_in_video2": len(new_in_v2),
        "reidentification_rate": (len(reidentified) / len(v1_ids)) if v1_ids else None,
        "reidentified_object_ids": [
            {
                "object_id": oid, "class_name": registry[oid].class_name,
                "first_seen_video1_frame": registry[oid].first_frame_id,
                "first_reidentified_video2_frame": registry[oid].v2_first_frame_id,
                "video1_hits": registry[oid].v1_hits, "video2_hits": registry[oid].v2_hits,
            }
            for oid in reidentified
        ],
        "missed_object_ids": [
            {"object_id": oid, "class_name": registry[oid].class_name, "video1_hits": registry[oid].v1_hits}
            for oid in missed
        ],
        "new_object_ids_in_video2": [
            {"object_id": oid, "class_name": registry[oid].class_name, "video2_hits": registry[oid].v2_hits}
            for oid in new_in_v2
        ],
        "total_seconds": float(total_seconds),
        "output_dir": str(output_dir),
    }
    report_path = output_dir / "reid_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print("[REMIND-REID] Done.")
    print(f"[REMIND-REID] Objects catalogued in video1: {len(v1_ids)}")
    rate_str = f"{report['reidentification_rate']*100:.1f}%" if report["reidentification_rate"] is not None else "n/a"
    print(f"[REMIND-REID] Re-identified in video2: {len(reidentified)}/{len(v1_ids)} ({rate_str})")
    print(f"[REMIND-REID] Missed (not seen again): {len(missed)}")
    print(f"[REMIND-REID] New identities created in video2: {len(new_in_v2)}")
    print(f"[REMIND-REID] Video1 rendered: {output_dir / 'video1_tracking.mp4'}")
    print(f"[REMIND-REID] Video2 rendered: {output_dir / 'video2_tracking.mp4'}")
    print(f"[REMIND-REID] Report: {report_path}")


if __name__ == "__main__":
    main()
