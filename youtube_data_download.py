
from youtube_transcript_api import YouTubeTranscriptApi
import random


# Function to fetch transcript from YouTube
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript


# Function to assign speakers heuristically (if no diarization model is used)
def assign_speakers(transcript):
    speakers = ["Alice", "Bob", "Charlie", "Moderator"]  # Example speakers
    speaker_transcript = []

    for i, entry in enumerate(transcript):
        speaker = speakers[i % len(speakers)]  # Assign speakers in a round-robin manner
        speaker_transcript.append(f"{speaker}: {entry['text']}")

    return "\n".join(speaker_transcript)


# Example usage
video_id="SvbAH6dDNE4"  # Replace with the actual YouTube video ID
transcript = get_transcript(video_id)

# Assign speakers heuristically
formatted_transcript = assign_speakers(transcript)
print(formatted_transcript)

# You can save the transcript to a text file
with open("transcript_with_speakers.txt", "w") as f:
    f.write(formatted_transcript)
