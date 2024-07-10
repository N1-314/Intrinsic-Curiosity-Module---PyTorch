import ffmpeg
import numpy as np
from PIL import Image


def array_to_image(arr):
    if arr.ndim == 3 and arr.shape[2] == 1:   # (w,h,1)
        arr = arr.squeeze(axis=-1)

    img = Image.fromarray(arr.astype('uint8'))
    return img


def create_video(frames, output_filename, fps=30, is_color=True):
    """
    Create a video from a list of NumPy arrays in the form of (height, width, 3).
    
    :param frames: a list of NumPy arrays in the form of (height, width, 3)
    :param output_filename: the name of the output video file (for example, 'output.mp4')
    :param fps: frames per second (default: 30)

    """
    # Check shape of the first frame
    sample_frame = frames[0]
    if sample_frame.ndim == 2:
        height, width = sample_frame.shape
    else:
        height, width = frames[0].shape[:2]

    # Determine pixel format
    if is_color:
        pix_fmt_in = 'rgb24'
    else:
        pix_fmt_in = 'gray'
    
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt=pix_fmt_in, s=f'{width}x{height}', r=fps)
        .output(output_filename, pix_fmt='yuv420p', loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    
    for frame in frames:
        if frame.ndim == 2:
            # expand (h,w) to (h,w,1)
            frame = np.expand_dims(frame, axis=-1)
        
        if not is_color and frame.shape[-1] == 1:
            # remove the last dimension for the channel
            frame = frame.squeeze(axis=-1)
        elif is_color and frame.shape[-1] == 1:
            # grayscale to RGB
            frame = np.repeat(frame, 3, axis=-1)
        
        process.stdin.write(frame.astype(np.uint8).tobytes())
    
    process.stdin.close()
    process.wait()

    print(f"A video has been created: {output_filename}", flush=True)