import ffmpeg
import numpy as np
from PIL import Image


def array_to_image(arr):
    img = Image.fromarray(arr.astype('uint8'))
    return img


def create_video(frames, output_filename, fps=30):
    """
    Create a video from a list of NumPy arrays in the form of (height, width, 3).
    
    :param frames: a list of NumPy arrays in the form of (height, width, 3)
    :param output_filename: the name of the output video file (for example, 'output.mp4')
    :param fps: frames per second (default: 30)

    """
    height, width = frames[0].shape[:2]
    
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=fps)
        .output(output_filename, pix_fmt='yuv420p', loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    
    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    
    process.stdin.close()
    process.wait()

    print(f"A video has been created: {output_filename}", flush=True)