#!/usr/bin/env python3
"""Test script for multi-anchor collision detection pipeline."""

import os
import sys

# Add to path
sys.path.append(os.path.dirname(__file__))

from examples.trajectory_demo.collision_detection_pipeline_yolo_first_method_a import YOLOFirstPipelineA


def main():
    # Use a test video
    video_path = "./videos/HomographTest_5s.mp4"

    # Use existing homography calibration
    homography_path = "./calibration/homographTest_5s_homography.json"

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    if not os.path.exists(homography_path):
        print(f"Homography not found: {homography_path}")
        print("Running without homography transform")
        homography_path = None

    print("=" * 70)
    print("Testing Multi-Anchor Collision Detection Pipeline")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    print(f"Homography: {homography_path}")

    # Initialize pipeline
    pipeline = YOLOFirstPipelineA(
        video_path=video_path,
        homography_path=homography_path,
        output_base="/workspace/ultralytics/results",
        skip_frames=1,
        model="yolo11n",
    )

    # Load homography if available
    if homography_path:
        pipeline.load_homography()

    # Step 1: Run YOLO detection
    print("\n【Step 1: YOLO Detection】")
    all_detections = pipeline.run_yolo_detection(conf_threshold=0.45)

    # Step 1.5: Merge fragmented objects
    print("\n【Step 1.5: Merge Fragmented Objects】")
    all_detections = pipeline.merge_fragmented_objects_in_frame(all_detections)

    # Step 2: Build trajectories
    print("\n【Step 2: Build Trajectories】")
    tracks = pipeline.build_trajectories(all_detections)

    # Step 3: Extract key frames with MULTI-ANCHOR collision detection
    print("\n【Step 3: Extract Key Frames (Multi-Anchor Collision Detection)】")
    proximity_events = pipeline.extract_key_frames(
        all_detections, tracks, world_distance_threshold=2.0, debug_threshold=5.0
    )

    # Step 4: Homography transform (if available)
    if pipeline.H is not None:
        print("\n【Step 4: Homography Transform】")
        transformed_events = pipeline.apply_homography_transform(proximity_events, tracks)
    else:
        print("\n【Step 4: Homography Transform】(Skipped - no homography)")
        transformed_events = proximity_events

    # Step 5: Analyze collision risk
    print("\n【Step 5: Collision Risk Analysis】")
    analyzed_events, level_counts = pipeline.analyze_collision_risk(transformed_events)

    # Generate report
    print("\n【Step 6: Generate Report】")
    pipeline.generate_report(proximity_events, analyzed_events, level_counts)

    print("\n" + "=" * 70)
    print(f"Pipeline complete! Results in: {pipeline.run_dir}")
    print("=" * 70)

    # Print summary of changes
    print("\n✨ MULTI-ANCHOR COLLISION DETECTION IMPROVEMENTS:")
    print("-" * 70)
    if proximity_events:
        event = proximity_events[0]
        if "closest_parts" in event:
            print("✓ Event now includes 'closest_parts' information:")
            print(f"  - Object 1 part: {event['closest_parts'].get('object1_part', 'N/A')}")
            print(f"  - Object 2 part: {event['closest_parts'].get('object2_part', 'N/A')}")
            print(f"  - Min distance: {event['closest_parts'].get('min_distance_px', 'N/A'):.1f} px")

        if "heading_analysis" in event:
            print("✓ Event now includes 'heading_analysis':")
            print(f"  - Relative heading: {event['heading_analysis'].get('relative_heading_rad', 'N/A'):.2f} rad")
            print(f"  - Approach direction: {event['heading_analysis'].get('approach_direction', 'N/A')}")

        if "ttc_seconds" in event:
            ttc = event.get("ttc_seconds")
            if ttc is not None:
                print(f"✓ Event now includes 'ttc_seconds': {ttc:.2f}s")
            else:
                print("✓ Event now includes 'ttc_seconds': N/A (objects separating)")

        if "risk_level" in event:
            print(f"✓ Enhanced risk level: {event.get('risk_level', 'N/A')}")

    print("\n📊 VISUALIZATION STATUS:")
    print("-" * 70)
    keyframe_dir = pipeline.keyframe_dir
    if keyframe_dir.exists():
        keyframes = list(keyframe_dir.glob("keyframe_*.jpg"))
        print(f"✓ Generated {len(keyframes)} keyframe images with:")
        print("  - Anchor points visualization")
        print("  - Closest parts highlighted")
        print("  - Distance and part names labeled")
        print("  - Risk level color-coded")

        if keyframes:
            print(f"\n✓ Keyframes saved in: {keyframe_dir}")
            print(f"  Example: {keyframes[0].name}")
    else:
        print("✗ No keyframes generated (check if events were detected)")


if __name__ == "__main__":
    main()
