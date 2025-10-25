"""
Gradio app for visualizing and comparing video stabilization results.

Allows side-by-side comparison of original vs stabilized frames,
with support for comparing multiple models.

Usage:
    python app.py
    python app.py --port 7860
    python app.py --share  # Create public link
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
import time


def load_data_split(split_path):
    """Load data split configuration."""
    if not Path(split_path).exists():
        return None
    with open(split_path, 'r') as f:
        return json.load(f)


def get_available_models(output_dir):
    """Get list of available stabilization models."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    # Exclude special directories that are not models
    exclude_dirs = {'videos', 'vis', 'cache', 'temp'}

    models = []
    for model_dir in output_path.iterdir():
        if model_dir.is_dir() and model_dir.name not in exclude_dirs:
            models.append(model_dir.name)

    return sorted(models)


def load_evaluation_results(output_dir, model_name):
    """
    Load evaluation results for a model if available.

    Args:
        output_dir: Path to output directory
        model_name: Name of the model

    Returns:
        dict or None: Evaluation results if available
    """
    if model_name == "Raw":
        # Try to load original metrics
        original_metrics_path = Path(output_dir) / "original_metrics.json"
        if not original_metrics_path.exists():
            return None

        try:
            with open(original_metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading original metrics: {e}")
            return None

    eval_path = Path(output_dir) / model_name / "evaluation_results.json"
    if not eval_path.exists():
        return None

    try:
        with open(eval_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {eval_path}")
        print(f"  Line {e.lineno}, Column {e.colno}: {e.msg}")
        print(f"  Please re-run: python src/evaluate.py --model {model_name}")
        return None
    except Exception as e:
        print(f"Error loading evaluation results for {model_name}: {e}")
        return None


def format_metrics_display(eval_results, model_name, current_flight=None):
    """
    Format evaluation metrics for display in the UI.

    Args:
        eval_results: Evaluation results dict (or None)
        model_name: Name of the model
        current_flight: Optional current flight name to highlight per-flight metrics

    Returns:
        str: Formatted markdown text
    """
    if eval_results is None:
        return f"**{model_name}:** _No evaluation metrics available_"

    agg = eval_results.get('aggregate_metrics', {})
    per_flight = eval_results.get('per_flight_metrics', [])

    # Check if this is original metrics (no stabilization)
    if model_name == "Raw" or eval_results.get('model_name') == 'original':
        # Format original-only metrics
        avg_diff = agg.get('avg_interframe_diff', 0)
        avg_flow = agg.get('avg_flow_magnitude', 0)
        total_frames = agg.get('total_frames', 0)
        num_flights = agg.get('num_flights', len(per_flight))

        metrics_text = f"""**{model_name} - Original Video Metrics:**
"""

        # Add per-flight metrics first
        if per_flight and current_flight:
            # Find current flight metrics
            flight_metrics = next((m for m in per_flight if m['flight_name'] == current_flight), None)
            if flight_metrics:
                metrics_text += f"""
**Current Flight ({current_flight}):**
- **Frames:** {flight_metrics.get('num_frames_original', 0)}
- **Inter-frame Diff:** {flight_metrics.get('original_avg_interframe_diff', 0):.2f}
- **Flow Magnitude:** {flight_metrics.get('original_avg_flow_magnitude', 0):.2f}

"""

        # Add aggregate metrics
        metrics_text += f"""**Aggregate ({num_flights} flights, {total_frames} frames):**
- **Avg Inter-frame Difference:** {avg_diff:.2f}
- **Avg Optical Flow Magnitude:** {avg_flow:.2f}

_Baseline metrics for unstabilized video_"""

        return metrics_text

    # Extract stabilization metrics
    diff_improv = agg.get('avg_improvement_interframe_diff', 0)
    flow_improv = agg.get('avg_improvement_flow_magnitude', 0)
    psnr = agg.get('avg_psnr', 0)
    sharpness = agg.get('avg_sharpness', 0)
    crop_ratio = agg.get('avg_cropping_ratio', 0)
    num_flights = eval_results.get('num_flights', len(per_flight))

    # Build formatted string
    metrics_text = f"""**{model_name} - Evaluation Metrics:**
"""

    # Add per-flight metrics for current flight first
    if per_flight and current_flight:
        flight_metrics = next((m for m in per_flight if m['flight_name'] == current_flight), None)
        if flight_metrics:
            metrics_text += f"""
**Current Flight ({current_flight}):**
- **Frames:** {flight_metrics.get('num_frames_stabilized', 0)}
- **Diff Improvement:** {flight_metrics.get('improvement_interframe_diff', 0):.1f}%
- **Flow Improvement:** {flight_metrics.get('improvement_flow_magnitude', 0):.1f}%
- **PSNR:** {flight_metrics.get('stabilized_avg_psnr', 0):.2f} dB
- **Crop Ratio:** {flight_metrics.get('stabilized_avg_cropping_ratio', 0):.1%}

"""

    # Add aggregate metrics
    metrics_text += f"""**Aggregate ({num_flights} flights):**
- **Inter-frame Diff Improvement:** {diff_improv:.1f}%
- **Flow Magnitude Improvement:** {flow_improv:.1f}%
- **Avg PSNR:** {psnr:.2f} dB
- **Avg Sharpness:** {sharpness:.2f}
- **Avg Cropping Ratio:** {crop_ratio:.1%}
"""

    # Add advanced metrics if available
    if 'avg_stability_score_fft' in agg:
        metrics_text += f"- **Stability Score (FFT):** {agg['avg_stability_score_fft']:.4f}\n"
    if 'avg_distortion_score' in agg:
        metrics_text += f"- **Avg Distortion Score:** {agg['avg_distortion_score']:.4f}\n"

    return metrics_text


def get_test_frames(data_dir, split_config):
    """
    Get test set frames organized by flight.

    Returns:
        dict: {flight_name: [(frame_idx, frame_path), ...]}
    """
    if split_config is None:
        # If no split, return all frames
        data_path = Path(data_dir)
        all_frames = {}
        for flight_dir in sorted(data_path.iterdir()):
            if flight_dir.is_dir():
                frames = []
                for img_path in sorted(flight_dir.glob("*.png")) + sorted(flight_dir.glob("*.jpg")):
                    try:
                        frame_idx = int(img_path.stem)
                        frames.append((frame_idx, str(img_path)))
                    except ValueError:
                        continue
                if frames:
                    all_frames[flight_dir.name] = sorted(frames)
        return all_frames

    # Get test set frames only
    test_splits = split_config['splits']['test']
    data_path = Path(data_dir)

    test_frames = {}
    for flight_name, frame_indices in test_splits.items():
        if not frame_indices:
            continue

        flight_dir = data_path / flight_name
        if not flight_dir.exists():
            continue

        frames = []
        for frame_idx in sorted(frame_indices):
            # Try different extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                frame_path = flight_dir / f"{frame_idx:08d}{ext}"
                if frame_path.exists():
                    frames.append((frame_idx, str(frame_path)))
                    break

        if frames:
            test_frames[flight_name] = frames

    return test_frames


def get_stabilized_frame(output_dir, model_name, flight_name, frame_idx):
    """Get stabilized frame for a specific model, flight, and frame index."""
    if model_name == "Raw":
        return None

    model_path = Path(output_dir) / model_name / flight_name
    if not model_path.exists():
        return None

    # Try different extensions
    for ext in ['.jpg', '.png', '.jpeg']:
        frame_path = model_path / f"{frame_idx:08d}{ext}"
        if frame_path.exists():
            return str(frame_path)

    return None


def load_and_prepare_image(image_path, max_size=800):
    """Load image and resize if needed."""
    if image_path is None:
        # Return placeholder
        placeholder = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Frame not available", (max_size//4, max_size//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder

    img = cv2.imread(image_path)
    if img is None:
        # Return error placeholder
        placeholder = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Failed to load", (max_size//4, max_size//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    return img


class StabilizationViewer:
    """Main viewer class for the Gradio app."""

    def __init__(self, data_dir='data/images', output_dir='output', split_path='data/data_split.json'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.split_path = split_path

        # Load data
        self.split_config = load_data_split(split_path)
        self.test_frames = get_test_frames(data_dir, self.split_config)
        self.available_models = get_available_models(output_dir)

        # State
        self.current_flight = None
        self.current_frame_idx = 0

        # Playback state
        self.is_playing = False

        print(f"Loaded {len(self.test_frames)} flights")
        print(f"Available models: {self.available_models}")

    def get_flight_names(self):
        """Get list of flight names."""
        return sorted(self.test_frames.keys())

    def get_model_options(self):
        """Get model options including 'Raw'."""
        return ["Raw"] + self.available_models

    def get_frame_count(self, flight_name):
        """Get number of frames for a flight."""
        if flight_name not in self.test_frames:
            return 0
        return len(self.test_frames[flight_name])

    def get_frame_info(self, flight_name, frame_position):
        """Get frame info for a specific position in the flight."""
        if flight_name not in self.test_frames:
            return None, None

        frames = self.test_frames[flight_name]
        if frame_position >= len(frames):
            return None, None

        frame_idx, frame_path = frames[frame_position]
        return frame_idx, frame_path

    def compare_frames(self, flight_name, frame_position, left_model, right_model):
        """
        Compare two frames side by side.

        Args:
            flight_name: Name of the flight
            frame_position: Position in the frame sequence (0-indexed)
            left_model: Model for left side ("Raw" or model name)
            right_model: Model for right side ("Raw" or model name)

        Returns:
            tuple: (left_image, right_image, info_text)
        """
        if flight_name not in self.test_frames:
            return None, None, "Flight not found"

        frames = self.test_frames[flight_name]
        if frame_position >= len(frames):
            return None, None, "Frame index out of range"

        frame_idx, raw_frame_path = frames[frame_position]

        # Get left image
        if left_model == "Raw":
            left_path = raw_frame_path
        else:
            left_path = get_stabilized_frame(self.output_dir, left_model, flight_name, frame_idx)

        # Get right image
        if right_model == "Raw":
            right_path = raw_frame_path
        else:
            right_path = get_stabilized_frame(self.output_dir, right_model, flight_name, frame_idx)

        # Load images
        left_img = load_and_prepare_image(left_path)
        right_img = load_and_prepare_image(right_path)

        # Create info text
        info = f"**Flight:** {flight_name} | **Frame:** {frame_idx} | **Position:** {frame_position + 1}/{len(frames)}"

        return left_img, right_img, info


def create_app(data_dir='data/images', output_dir='output', split_path='data/data_split.json'):
    """Create and configure the Gradio app."""

    viewer = StabilizationViewer(data_dir, output_dir, split_path)

    if not viewer.test_frames:
        print("Warning: No frames found. Please check your data directory.")

    with gr.Blocks(title="Video Stabilization Viewer") as app:
        gr.Markdown("""
        # 🎥 Video Stabilization Viewer

        Compare original and stabilized video frames side by side. Select different models to evaluate stabilization quality.
        """)

        # Top controls: Flight selection and navigation
        with gr.Row():
            flight_dropdown = gr.Dropdown(
                choices=viewer.get_flight_names(),
                label="Flight",
                value=viewer.get_flight_names()[0] if viewer.get_flight_names() else None,
                interactive=True,
                scale=2
            )

            frame_slider = gr.Slider(
                minimum=0,
                maximum=viewer.get_frame_count(viewer.get_flight_names()[0]) - 1 if viewer.get_flight_names() else 0,
                step=1,
                value=0,
                label="Frame Position",
                interactive=True,
                scale=2
            )

        # Info text
        with gr.Row():
            info_text = gr.Markdown("Select a flight and frame to compare", elem_classes="info-text")

        # Playback controls: all buttons on one line
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    prev_btn = gr.Button("◀", size="sm", min_width=50)
                    play_btn = gr.Button("▶ Play", size="sm", variant="primary", min_width=80)
                    pause_btn = gr.Button("⏸ Pause", size="sm", variant="secondary", min_width=80, interactive=False)
                    next_btn = gr.Button("▶", size="sm", min_width=50)

            with gr.Column(scale=1):
                with gr.Row():
                    gr.Markdown("**Playback FPS:**", elem_classes="inline-label")
                    fps_number = gr.Number(
                        value=30,
                        label="",
                        minimum=1,
                        maximum=30,
                        precision=0,
                        scale=1,
                        container=False
                    )

                    loop_checkbox = gr.Checkbox(
                        label="Loop",
                        value=True,
                        interactive=True,
                        scale=1
                    )

                    show_videos = gr.Checkbox(
                        label="Show videos (if available)",
                        value=False,
                        interactive=True,
                        scale=1
                    )
        # Main comparison view: dropdowns above their respective images
        with gr.Row():
            with gr.Column():
                left_model = gr.Dropdown(
                    choices=viewer.get_model_options(),
                    label="Left Side Model",
                    value="Raw",
                    interactive=True
                )
                left_image = gr.Image(label=None, type="numpy", visible=True)
                left_video = gr.Video(label=None, visible=False, autoplay=True)
                left_metrics = gr.Markdown("", elem_classes="metrics-text")

            with gr.Column():
                right_model = gr.Dropdown(
                    choices=viewer.get_model_options(),
                    label="Right Side Model",
                    value=viewer.available_models[0] if viewer.available_models else "Raw",
                    interactive=True
                )
                right_image = gr.Image(label=None, type="numpy", visible=True)
                right_video = gr.Video(label=None, visible=False, autoplay=True)
                right_metrics = gr.Markdown("", elem_classes="metrics-text")

        # Event handlers
        def update_slider_range(flight_name):
            """Update slider range when flight changes."""
            count = viewer.get_frame_count(flight_name)
            return gr.Slider(maximum=max(0, count - 1), value=0)

        def _get_video_path_for_model(model_name, flight_name):
            """Return absolute path to pre-rendered video for a model+flight if it exists."""
            out_videos = Path(viewer.output_dir) / "videos"
            if model_name == "Raw":
                p = out_videos / "original" / f"{flight_name}.mp4"
                if p.exists():
                    return str(p.absolute())
                return None

            # model_name may match a folder under output/videos (e.g., baseline, fusion)
            candidate = out_videos / model_name / f"{flight_name}.mp4"
            if candidate.exists():
                return str(candidate.absolute())

            return None


        def update_comparison(flight_name, frame_pos, left_mod, right_mod, use_videos):
            """Update images/videos and metrics when any parameter changes."""
            # Prepare metrics
            left_eval = load_evaluation_results(viewer.output_dir, left_mod)
            right_eval = load_evaluation_results(viewer.output_dir, right_mod)

            left_metrics_text = format_metrics_display(left_eval, left_mod, flight_name)
            right_metrics_text = format_metrics_display(right_eval, right_mod, flight_name)

            info = f"**Flight:** {flight_name} | **Position:** {int(frame_pos) + 1}/{viewer.get_frame_count(flight_name)}"

            if use_videos:
                # Try to find videos for left and right
                left_vid = _get_video_path_for_model(left_mod, flight_name)
                right_vid = _get_video_path_for_model(right_mod, flight_name)

                # Debug output
                print(f"[VIDEO MODE] Left: {left_mod} -> {left_vid}")
                print(f"[VIDEO MODE] Right: {right_mod} -> {right_vid}")

                # Update info to indicate video mode
                video_status = []
                if left_vid:
                    video_status.append("Left: ✓")
                else:
                    video_status.append("Left: ✗ (no video)")
                if right_vid:
                    video_status.append("Right: ✓")
                else:
                    video_status.append("Right: ✗ (no video)")

                info = f"**Flight:** {flight_name} | **Video Mode** | {' | '.join(video_status)}"

                # Hide image components and show video components with proper paths
                return (
                    gr.update(visible=False),  # left_image
                    gr.update(value=left_vid, visible=True) if left_vid else gr.update(visible=True),  # left_video
                    gr.update(visible=False),  # right_image
                    gr.update(value=right_vid, visible=True) if right_vid else gr.update(visible=True),  # right_video
                    info,  # info_text
                    left_metrics_text,  # left_metrics
                    right_metrics_text  # right_metrics
                )

            # Otherwise show images as before
            frame_idx, raw_frame_path = viewer.get_frame_info(flight_name, int(frame_pos))
            left_path = raw_frame_path if left_mod == "Raw" else get_stabilized_frame(viewer.output_dir, left_mod, flight_name, frame_idx)
            right_path = raw_frame_path if right_mod == "Raw" else get_stabilized_frame(viewer.output_dir, right_mod, flight_name, frame_idx)

            left_img = load_and_prepare_image(left_path)
            right_img = load_and_prepare_image(right_path)

            # Ensure image components visible and videos hidden
            left_img_update = gr.update(value=left_img, visible=True)
            right_img_update = gr.update(value=right_img, visible=True)
            left_vid_update = gr.update(visible=False)
            right_vid_update = gr.update(visible=False)

            return left_img_update, left_vid_update, right_img_update, right_vid_update, info, left_metrics_text, right_metrics_text

        def prev_frame(frame_pos):
            """Go to previous frame."""
            new_pos = max(0, frame_pos - 1)
            return new_pos

        def next_frame(flight_name, frame_pos):
            """Go to next frame."""
            max_pos = viewer.get_frame_count(flight_name) - 1
            new_pos = min(max_pos, frame_pos + 1)
            return new_pos

        # Playback state tracking
        playback_control = {"should_stop": False}

        def start_playback(flight_name, frame_pos, fps, loop, left_mod, right_mod, use_videos):
            """Start video playback. If use_videos is True, return the video paths and skip frame streaming."""
            playback_control["should_stop"] = False
            max_pos = viewer.get_frame_count(flight_name) - 1
            current_pos = int(frame_pos)

            # Load metrics once at start
            left_eval = load_evaluation_results(viewer.output_dir, left_mod)
            right_eval = load_evaluation_results(viewer.output_dir, right_mod)
            left_metrics_text = format_metrics_display(left_eval, left_mod, flight_name)
            right_metrics_text = format_metrics_display(right_eval, right_mod, flight_name)

            # If using videos, return video sources and do not stream frames
            if use_videos:
                # Use the existing helper function for consistency
                left_vid = _get_video_path_for_model(left_mod, flight_name)
                right_vid = _get_video_path_for_model(right_mod, flight_name)

                # Hide images, show videos if available
                left_img_update = gr.update(visible=False)
                right_img_update = gr.update(visible=False)
                left_vid_update = gr.update(value=left_vid, visible=bool(left_vid))
                right_vid_update = gr.update(value=right_vid, visible=bool(right_vid))

                # Return a single update (metrics unchanged)
                yield (
                    left_img_update,
                    left_vid_update,
                    right_img_update,
                    right_vid_update,
                    f"**Flight:** {flight_name}",
                    gr.update(value=int(frame_pos)),
                    gr.update(value=left_metrics_text),
                    gr.update(value=right_metrics_text),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                )

                return

            # Otherwise perform frame-by-frame streaming as before
            # Enable pause button, disable play button
            yield (
                gr.update(),
                gr.update(visible=False),
                gr.update(),
                gr.update(visible=False),
                gr.update(),
                gr.update(value=int(frame_pos)),
                gr.update(),
                gr.update(),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )

            # Play through frames
            while not playback_control["should_stop"]:
                # Get current frame
                left_img, right_img, info = viewer.compare_frames(
                    flight_name, current_pos, left_mod, right_mod
                )

                # Yield updated frame (metrics stay the same during playback)
                yield (
                    left_img,
                    gr.update(visible=False),
                    right_img,
                    gr.update(visible=False),
                    info,
                    int(current_pos),
                    left_metrics_text,
                    right_metrics_text,
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                )

                # Sleep based on FPS
                time.sleep(1.0 / fps)

                # Move to next frame
                current_pos += 1

                # Loop back if enabled
                if current_pos > max_pos:
                    if loop:
                        current_pos = 0
                    else:
                        # Reached end, stop playing
                        break

            # Playback ended, re-enable play button, disable pause button
            yield (
                gr.update(),
                gr.update(visible=False),
                gr.update(),
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        def stop_playback():
            """Stop video playback."""
            playback_control["should_stop"] = True
            return gr.update(interactive=True), gr.update(interactive=False)

        # Connect events
        flight_dropdown.change(
            fn=update_slider_range,
            inputs=[flight_dropdown],
            outputs=[frame_slider]
        )

        # Update comparison when any control changes (include show_videos)
        for component in [flight_dropdown, frame_slider, left_model, right_model, show_videos]:
            component.change(
                fn=update_comparison,
                inputs=[flight_dropdown, frame_slider, left_model, right_model, show_videos],
                outputs=[left_image, left_video, right_image, right_video, info_text, left_metrics, right_metrics]
            )

        # Navigation buttons
        prev_btn.click(
            fn=prev_frame,
            inputs=[frame_slider],
            outputs=[frame_slider]
        )

        next_btn.click(
            fn=next_frame,
            inputs=[flight_dropdown, frame_slider],
            outputs=[frame_slider]
        )

        # Play button with streaming
        play_btn.click(
            fn=start_playback,
            inputs=[
                flight_dropdown,
                frame_slider,
                fps_number,
                loop_checkbox,
                left_model,
                right_model,
                show_videos,
            ],
            outputs=[
                left_image,
                left_video,
                right_image,
                right_video,
                info_text,
                frame_slider,
                left_metrics,
                right_metrics,
                play_btn,
                pause_btn,
            ]
        )

        # Pause button
        pause_btn.click(
            fn=stop_playback,
            inputs=[],
            outputs=[play_btn, pause_btn]
        )

        # Load initial comparison (default to frame mode, not video mode)
        def initial_load():
            """Load initial state - always start with frame view."""
            flight = viewer.get_flight_names()[0] if viewer.get_flight_names() else None
            if not flight:
                return None, gr.update(visible=False), None, gr.update(visible=False), "No flights available", "", ""

            left_mod = "Raw"
            right_mod = viewer.available_models[0] if viewer.available_models else "Raw"

            # Load initial frame
            frame_idx, raw_frame_path = viewer.get_frame_info(flight, 0)
            left_path = raw_frame_path
            right_path = raw_frame_path if right_mod == "Raw" else get_stabilized_frame(viewer.output_dir, right_mod, flight, frame_idx)

            left_img = load_and_prepare_image(left_path)
            right_img = load_and_prepare_image(right_path)

            # Load metrics
            left_eval = load_evaluation_results(viewer.output_dir, left_mod)
            right_eval = load_evaluation_results(viewer.output_dir, right_mod)
            left_metrics_text = format_metrics_display(left_eval, left_mod, flight)
            right_metrics_text = format_metrics_display(right_eval, right_mod, flight)

            info = f"**Flight:** {flight} | **Frame:** {frame_idx} | **Position:** 1/{viewer.get_frame_count(flight)}"

            return (
                gr.update(value=left_img, visible=True),
                gr.update(visible=False),
                gr.update(value=right_img, visible=True),
                gr.update(visible=False),
                info,
                left_metrics_text,
                right_metrics_text
            )

        app.load(
            fn=initial_load,
            inputs=[],
            outputs=[left_image, left_video, right_image, right_video, info_text, left_metrics, right_metrics]
        )

        gr.Markdown("""
        ---
        ### Tips
        - **Single Frame**: Use the slider or ◀ ▶ buttons to browse frames
        - **Video Playback**: Click "▶ Play" to play through frames as a video
        - **Rendered Videos**: Check "Show videos (if available)" to display pre-rendered videos from `output/videos/`
          - Videos must be in: `output/videos/original/` for Raw, `output/videos/{model}/` for models
          - Video format: `{FlightName}.mp4`
        - **Playback Speed**: Adjust FPS (1-30) to control playback speed
        - **Loop**: Enable to continuously loop through frames
        - **Model Comparison**: Select different models in the dropdowns above each image
        - **Raw vs Stabilized**: Select "Raw" to show the original unstabilized frame
        - **Evaluation Metrics**: Metrics appear below each image if available
          - For models: run `python src/evaluate.py --model <name>`
          - For "Raw": run `python src/evaluate.py --model original --split-set test`
        - The viewer shows only test set frames (from `data/data_split.json`)
        """)

    return app


def main():
    parser = argparse.ArgumentParser(description="Launch Video Stabilization Viewer")
    parser.add_argument('--data-dir', type=str, default='data/images',
                       help='Path to original images directory (default: data/images)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Path to output directory with stabilized results (default: output)')
    parser.add_argument('--split', type=str, default='data/data_split.json',
                       help='Path to data split JSON file (default: data/data_split.json)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the app on (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Create a public shareable link')
    parser.add_argument('--server-name', type=str, default='127.0.0.1',
                       help='Server name (default: 127.0.0.1, use 0.0.0.0 for external access)')

    args = parser.parse_args()

    # Get repository root
    repo_root = Path(__file__).parent
    data_dir = repo_root / args.data_dir
    output_dir = repo_root / args.output_dir
    split_path = repo_root / args.split

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split file: {split_path}")
    print(f"Starting app on port {args.port}...")

    # Create and launch app
    app = create_app(str(data_dir), str(output_dir), str(split_path))
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
