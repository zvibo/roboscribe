"""Main script for audio transcription and cleanup."""
import argparse
import os
import warnings
from pathlib import Path
import torch
from datetime import datetime

from .config import TranscriptionConfig, CleanupConfig, CommandLineArgs
from .transcript_processor import TranscriptProcessor
from .text_utils import split_long_lines

def parse_arguments() -> CommandLineArgs:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RoboScribe - automatic podcast transcript producer."
    )
    parser.add_argument(
        "--speakers",
        type=int,
        required=True,
        help="Number of speakers to use for diarization."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file. Should be a file in WAV format."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="HuggingFace token. Using a read-only token is acceptable."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output text file. There will be two files generated - a raw file, and a cleaned up version."
    )
    args = parser.parse_args()
    return CommandLineArgs(**vars(args))

def get_transcription_config() -> TranscriptionConfig:
    """Create transcription configuration."""
    return TranscriptionConfig(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def process_transcript(
    args: CommandLineArgs,
    processor: TranscriptProcessor
) -> None:
    """Process the transcript and save results."""
    raw_output_file = Path(args.output_file).with_suffix('.raw.txt')
    
    if raw_output_file.exists():
        use_existing = input(
            f"{raw_output_file} already exists. Use it for cleanup? (y/n): "
        ).strip().lower() == 'y'
    else:
        use_existing = False

    if not use_existing:
        # Process audio and save raw output
        text_segments = processor.process_audio(args.audio_path, args.speakers)
        text_segments = split_long_lines(text_segments)
        
        print(f"Saving raw output to {raw_output_file}...")
        raw_output_file.write_text(
            '\n'.join(text_segments) + '\n',
            encoding='utf-8'
        )
    else:
        # Load existing raw output
        print(f"Loading raw output from {raw_output_file}...")
        text_segments = raw_output_file.read_text(
            encoding='utf-8'
        ).splitlines()

    # Clean and save processed output
    print(f"\nStarting transcript cleanup")
    print(f"Total lines to process: {len(text_segments)}")
    
    cleaned_lines = []
    for idx, line in enumerate(text_segments):
        cleaned_text = processor.clean_transcript_segment(
            line.strip(),
            idx,
            len(text_segments)
        )
        cleaned_lines.append(cleaned_text)
        
        # Save progress periodically
        if (idx + 1) % 10 == 0:
            temp_output_file = Path(args.output_file).with_suffix('.temp.txt')
            temp_output_file.write_text(
                '\n'.join(cleaned_lines) + '\n',
                encoding='utf-8'
            )
            print(f"Progress saved ({idx + 1}/{len(text_segments)} lines)")
    
    print(f"\nTranscript cleanup completed")
    print(f"Saving final output to {args.output_file}...")
    
    Path(args.output_file).write_text(
        '\n'.join(cleaned_lines) + '\n',
        encoding='utf-8'
    )

def main() -> None:
    """Main execution function."""
    warnings.filterwarnings("ignore")
    
    start_time = datetime.utcnow()
    print("Starting RoboScribe")
    
    args = parse_arguments()
    transcription_config = get_transcription_config()
    cleanup_config = CleanupConfig()
    
    processor = TranscriptProcessor(
        transcription_config,
        cleanup_config,
        args.hf_token
    )
    
    process_transcript(args, processor)
    
    end_time = datetime.utcnow()
    duration = end_time - start_time
    print("Processing complete")
    print(f"Total processing time: {duration}")

if __name__ == "__main__":
    main()
