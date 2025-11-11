import os
import whisper
import pandas as pd
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pathlib import Path


class AudioProcessor:
    """Handles audio file conversion and segment extraction"""
    
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audio = None
        
    def convert_to_wav(self, output_path="converted_audio.wav"):
        """Convert MP3 or other formats to WAV"""
        file_ext = Path(self.audio_path).suffix.lower()
        
        if file_ext == ".wav":
            self.audio = AudioSegment.from_wav(self.audio_path)
            return self.audio_path
        elif file_ext == ".mp3":
            sound = AudioSegment.from_mp3(self.audio_path)
            sound.export(output_path, format="wav")
            self.audio = sound
            return output_path
        else:
            # Try to load as general audio file
            sound = AudioSegment.from_file(self.audio_path)
            sound.export(output_path, format="wav")
            self.audio = sound
            return output_path
    
    def extract_segment(self, start_time, end_time, output_path):
        """Extract audio segment between start and end times (in seconds)"""
        if self.audio is None:
            raise ValueError("Audio not loaded. Call convert_to_wav first.")
        
        segment = self.audio[int(start_time * 1000):int(end_time * 1000)]
        segment.export(output_path, format="wav")
        return output_path


class SpeakerDiarizer:
    """Performs speaker diarization using pyannote.audio"""
    
    def __init__(self, huggingface_token):
        self.token = huggingface_token
        self.pipeline = None
        
    def load_pipeline(self):
        """Load the diarization pipeline"""
        print("Loading diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=self.token
        )
        print("Pipeline loaded successfully!")
        
    def diarize(self, audio_path):
        """Perform diarization on audio file"""
        if self.pipeline is None:
            self.load_pipeline()
        
        print("Running speaker diarization...")
        diarization = self.pipeline(audio_path)
        print("Diarization complete!")
        return diarization


class Transcriber:
    """Transcribes audio using OpenAI Whisper"""
    
    def __init__(self, model_size="large"):
        self.model_size = model_size
        self.model = None
        
    def load_model(self):
        """Load Whisper model"""
        print(f"Loading Whisper {self.model_size} model...")
        self.model = whisper.load_model(self.model_size)
        print("Whisper model loaded!")
        
    def transcribe(self, audio_path, language=None):
        """Transcribe audio file"""
        if self.model is None:
            self.load_model()
        
        result = self.model.transcribe(audio_path, language=language)
        return result["text"].strip()


class MeetingTranscriptGenerator:
    """Main class that orchestrates the entire transcription process"""
    
    def __init__(self, huggingface_token, whisper_model="large"):
        self.hf_token = huggingface_token
        self.whisper_model = whisper_model
        self.audio_processor = None
        self.diarizer = SpeakerDiarizer(huggingface_token)
        self.transcriber = Transcriber(whisper_model)
        
    def process_audio(self, audio_path, output_csv="meeting_transcript.csv", language=None):
        """
        Process audio file: diarize and transcribe
        
        Args:
            audio_path: Path to audio file
            output_csv: Output CSV filename
            language: Language code (e.g., 'ar' for Arabic, None for auto-detect)
        """
        # Initialize audio processor
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path}")
        print(f"{'='*60}\n")
        
        self.audio_processor = AudioProcessor(audio_path)
        
        # Convert to WAV if needed
        wav_path = self.audio_processor.convert_to_wav()
        print(f"Audio ready: {wav_path}\n")
        
        # Perform diarization
        diarization = self.diarizer.diarize(wav_path)
        
        # Load transcriber
        self.transcriber.load_model()
        
        # Process each segment
        results = []
        temp_files = []
        
        print("\nTranscribing segments...")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            
            # Extract segment
            segment_path = f"temp_{speaker}_{int(start)}.wav"
            self.audio_processor.extract_segment(start, end, segment_path)
            temp_files.append(segment_path)
            
            # Transcribe segment
            text = self.transcriber.transcribe(segment_path, language=language)
            
            results.append({
                "Speaker": speaker,
                "Start": round(start, 2),
                "End": round(end, 2),
                "Duration": round(end - start, 2),
                "Text": text
            })
            
            print(f"  [{speaker}] {start:.2f}s - {end:.2f}s: {text[:50]}...")
        
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        
        print(f"\n{'='*60}")
        print(f"✓ Transcript saved to: {output_csv}")
        print(f"✓ Total segments: {len(results)}")
        print(f"✓ Total speakers: {df['Speaker'].nunique()}")
        print(f"{'='*60}\n")
        
        return df


def main():
    """Main function to run the transcription process"""
    
    # Configuration
    HUGGINGFACE_TOKEN = "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"
    WHISPER_MODEL = "large"  # Options: "tiny", "base", "small", "medium", "large"
    
    print("\n" + "="*60)
    print("  AUDIO DIARIZATION & TRANSCRIPTION TOOL")
    print("="*60 + "\n")
    
    # Get audio file path from user
    audio_path = input("Enter the path to your audio file: ").strip()
    
    # Remove quotes if user copied path with quotes
    audio_path = audio_path.strip('"').strip("'")
    
    # Validate file exists
    if not os.path.exists(audio_path):
        print(f"\n❌ Error: File not found: {audio_path}")
        return
    
    # Get output filename
    default_output = "meeting_transcript.csv"
    output_csv = input(f"Enter output CSV filename (default: {default_output}): ").strip()
    if not output_csv:
        output_csv = default_output
    
    # Get language (optional)
    print("\nLanguage options:")
    print("  - Press Enter for auto-detection")
    print("  - 'ar' for Arabic")
    print("  - 'en' for English")
    print("  - Or any other Whisper-supported language code")
    language = input("Enter language code: ").strip() or None
    
    # Create generator and process
    generator = MeetingTranscriptGenerator(
        huggingface_token=HUGGINGFACE_TOKEN,
        whisper_model=WHISPER_MODEL
    )
    
    try:
        df = generator.process_audio(audio_path, output_csv, language)
        
        # Display summary
        print("\nTranscript Preview:")
        print("-" * 60)
        print(df.head(10).to_string(index=False))
        if len(df) > 10:
            print(f"\n... and {len(df) - 10} more segments")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()