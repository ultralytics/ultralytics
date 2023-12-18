import moviepy.editor as mp


def video2gif(video_path, gif_path, start_time, duration, fps):
    """Convert a video to gif.

    Args:
        video_path (str): Path to the video.
        gif_path (str): Path to the gif.
        start_time (float): Start time of the gif.
        duration (float): Duration of the gif.
        fps (int): Frames per second of the gif.
    """
    # get the video
    video = mp.VideoFileClip(video_path)
    # resize the video
    video = video.resize(0.5)
    # get the gif
    if duration is not None:
        gif = video.subclip(start_time, start_time + duration)
    else:
        gif = video.subclip(start_time)
    gif.speedx(3).to_gif(gif_path, fps=fps)


video2gif('./my_tracker/outputs/ByteTrack/annotated_video.mp4', './my_tracker/outputs/ByteTrack/annotated_video.gif', 0, None, 25)


# resize gif
def resize_gif(gif_path, new_gif_path, start_time, duration, fps):
    """Resize a gif.

    Args:
        gif_path (str): Path to the gif.
        new_gif_path (str): Path to the resized gif.
        scale (float): Scale of the resized gif.
    """
    # get the gif
    gif = mp.VideoFileClip(gif_path)
    # resize the gif
    # gif = gif.resize(0.5)
    # save the resized gif
    if duration is not None:
        gif = gif.subclip(start_time, start_time + duration)
    else:
        gif = gif.subclip(start_time)
    gif.speedx(3).to_gif(new_gif_path, fps=fps)


# resize_gif('./my_tracker/outputs/ByteTrack/annotated_video.mp4', './my_tracker/outputs/ByteTrack/annotated_video.gif', 0, None, 10)