import pprint
import tempfile
import time
import traceback

import streamlit as st

import analyzer


MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def process_video(video_bytes):
    try:
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4") as input_file,
            tempfile.NamedTemporaryFile(suffix=".webm") as output_file,
        ):
            input_file.write(video_bytes)
            task = analyzer.Analyzer(input_file.name, output_file.name, "VP80")
            task.run()

            video = output_file.read()
            stats = task.get_stats()

        return video, stats
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None


def analyze_video(upload):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading video...")
        progress_bar.progress(10)

        video_bytes = upload.getvalue()

        status_text.text("Processing video...")
        progress_bar.progress(30)

        video, stats = process_video(video_bytes)
        if video is None or stats is None:
            return

        progress_bar.progress(80)
        status_text.text("Displaying results...")

        col1.write("Processed Video :camera:")
        col1.video(video)

        col2.write("Tracker Data :wrench:")
        col2.code(
            pprint.pformat(stats),
            language="json",
        )

        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process video")
        # Log the full error for debugging.
        print(f"Error in fix_image: {traceback.format_exc()}")


st.set_page_config(layout="wide", page_title="People Tracker")

st.write("## People Tracker")
st.write(
    "Try uploading a video to track people in it. This code is open source and available [here](https://github.com/SirPersimmon/peopletracker) on GitHub. Special thanks to the [liveposetracker](https://github.com/ortegatron/liveposetracker)"
)
st.sidebar.write("## Upload :gear:")

# UI Layout.
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload a video", type=["mp4"])

# Information about limitations.
with st.sidebar.expander("ℹ️ Video Guidelines"):
    st.write(
        """
    - Maximum file size: 100MB
    - Supported formats: MP4
    - Processing time depends on video size
    """
    )

# Process the video.
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(
            f"The uploaded file is too large. Please upload a video smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB."
        )
    else:
        analyze_video(upload=my_upload)
else:
    st.info("Please upload a video to get started!")
