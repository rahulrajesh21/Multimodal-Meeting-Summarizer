import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from moviepy import ColorClip
from src.video_processing import VideoSummarizer

class MockScorer:
    def score_sentence(self, text, role):
        # Dummy score based on text content
        if "important" in text:
            return 0.9
        return 0.1

def test_summarizer():
    # 1. Create dummy MP4
    dummy_video_path = "tmp_dummy_video.mp4"
    if os.path.exists(dummy_video_path):
        os.remove(dummy_video_path)
        
    print("Generating dummy video...")
    clip = ColorClip(size=(640, 480), color=[255, 0, 0], duration=20).with_fps(24)
    clip.write_videofile(dummy_video_path, codec="libx264", audio=False, logger=None)

    scorer = MockScorer()
    summarizer = VideoSummarizer(highlight_scorer=scorer)

    segments = [
        {'start': 0.0, 'end': 3.0, 'text': 'This is the first important thing.', 'speaker': 'Developer'},
        {'start': 5.0, 'end': 8.0, 'text': 'I agree, we should do that soon.', 'speaker': 'Product Manager'},
        {'start': 10.0, 'end': 13.0, 'text': 'And here is the second important thing, the download button.', 'speaker': 'Developer'},
    ]

    print("Scoring segments...")
    scored = summarizer.score_segments(segments, "Developer")
    
    print("Filtering and smoothing...")
    time_ranges = summarizer.filter_and_smooth(scored, threshold=0.5, min_gap=2.0)
    print(f"Time ranges: {time_ranges}")
    
    output_path = "tmp_output_video.mp4"
    print("Creating summary video...")
    try:
        status = summarizer.create_summary_video(dummy_video_path, time_ranges, output_path)
        print(f"Status: {status}")
        if os.path.exists(output_path):
            print("Test passed! 🎉")
            os.remove(output_path)
    except Exception as e:
        print(f"Test failed: {e}")

    # Clean up
    if os.path.exists(dummy_video_path):
        os.remove(dummy_video_path)

if __name__ == "__main__":
    test_summarizer()
