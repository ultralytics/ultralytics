import subprocess


def convert_video_to_gif(video_path, gif_path, fps=25):
    """
    Convert a video file to a GIF.
    :param video_path: Path to the video file.
    :param gif_path: Path where the GIF should be saved.
    :param fps: Frames per second for the GIF.
    """
    """reader = imageio.get_reader(video_path)
    with imageio.get_writer(gif_path, fps=fps) as writer:
        for frame in reader:
            writer.append_data(frame)
    print(f"Saved GIF to {gif_path}")"""
    command = [
        'ffmpeg',
        '-i', video_path,  # Input file
        '-vf', f'fps={fps}',  # Set frame rate
        '-f', 'gif',  # Set format to GIF
        gif_path  # Output file
    ]
    subprocess.run(command, check=True)
    print(f"Saved GIF to {gif_path}")