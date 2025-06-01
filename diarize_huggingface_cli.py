#!/usr/bin/env python3
"""
Speaker diarization using pyannote with HuggingFace CLI authentication
This script uses CLI authentication which is more reliable than token-based auth
Make sure to run 'huggingface-cli login' before running this script
Uses pyannote/speaker-diarization-3.1 for improved performance
Transcription is performed using WhisperX.
"""

import gradio as gr
from pyannote.audio import Pipeline
from datetime import timedelta
import os
import sys
import subprocess
import time
import torch
import tempfile
import shutil
import torchaudio
import uuid
import threading
import glob
import json
from pathlib import Path
import pandas as pd

# Check for whisperx availability
WHISPERX_AVAILABLE = False
WHISPERX_VERSION = None

try:
    import whisperx
    import importlib.metadata
    try:
        WHISPERX_VERSION = importlib.metadata.version("whisperx")
        print(f"✅ WhisperX is available, version: {WHISPERX_VERSION}")
    except importlib.metadata.PackageNotFoundError:
        WHISPERX_VERSION = "unknown"
        print("✅ WhisperX is available, version: unknown")
    WHISPERX_AVAILABLE = True
except ImportError:
    print("❌ WhisperX is not available. This tool requires WhisperX for transcription.")
    print("   Please install it by ensuring it's in your requirements.txt and running:")
    print("   pip install -r requirements.txt")
    # Optionally, exit if WhisperX is critical and not found, depending on desired behavior
    # sys.exit("WhisperX not found. Please install it to use this application.")


# Configuration
DEFAULT_TEMP_DIR = os.path.join(tempfile.gettempdir(), "diarize_temp")
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Desktop")

# Create temp dir if it doesn't exist
os.makedirs(DEFAULT_TEMP_DIR, exist_ok=True)

# Global progress tracking
progress_status = {"status": "", "progress": 0.0}
progress_lock = threading.Lock()

# Language mapping from display name to Whisper code
LANGUAGE_NAME_TO_CODE_MAP = {
    "English": "en",
    "Chinese": "zh",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Korean": "ko",
    "French": "fr",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Turkish": "tr",
    "": None  # For auto-detect, maps to None
}

def update_progress(status, progress_value=None):
    """Update the global progress status"""
    with progress_lock:
        progress_status["status"] = status
        if progress_value is not None:
            progress_status["progress"] = float(progress_value)

def get_progress():
    """Get the current progress status"""
    with progress_lock:
        return progress_status.copy()

def check_hf_login():
    """Check if user is logged in to HuggingFace CLI"""
    update_progress("Checking Hugging Face login", 0.05)
    print("🔍 Checking Hugging Face CLI login status...")
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        username = result.stdout.strip()
        print(f"✅ Logged in to Hugging Face as: {username}")
        update_progress(f"Logged in as {username}", 0.1)
        return True
    except subprocess.CalledProcessError:
        print("❌ Not logged in to Hugging Face CLI")
        print("\n===================================================================")
        print("Please follow these steps:")
        print("1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   and click 'Access repository' to accept the license")
        print("2. Visit https://huggingface.co/pyannote/segmentation-3.0")
        print("   and click 'Access repository' to accept the license")
        print("3. Visit https://huggingface.co/settings/tokens to create a token")
        print("4. Run the command: huggingface-cli login")
        print("   and enter your token when prompted")
        print("===================================================================\n")
        update_progress("Not logged in to Hugging Face", 0)
        return False

def convert_audio_to_wav(input_file, temp_dir=DEFAULT_TEMP_DIR, trim_start=None, trim_end=None):
    """Convert audio file to WAV format (16kHz mono) using ffmpeg"""
    update_progress("Converting audio file", 0.15)
    print(f"🔄 Converting audio file to WAV format: {input_file}")
    
    # Create a temporary directory for the output file
    os.makedirs(temp_dir, exist_ok=True)
    temp_wav_path = os.path.join(temp_dir, f"temp_audio_{uuid.uuid4().hex}.wav")
    
    try:
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_file,
            "-ar", "16000",  # Sample rate: 16kHz
            "-ac", "1",      # Channels: 1 (mono)
            "-y",            # Overwrite output file if it exists
        ]
        
        # Add trim parameters if provided
        trimmed = False
        if trim_start is not None and trim_end is not None:
            if trim_start >= 0 and trim_end > trim_start:
                duration = trim_end - trim_start
                cmd.extend(["-ss", str(trim_start), "-t", str(duration)])
                trimmed = True
                print(f"🔪 Trimming audio from {trim_start}s to {trim_end}s (duration: {duration}s)")
        
        # Add output file
        cmd.append(temp_wav_path)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"❌ Error converting audio file: {result.stderr}")
            update_progress("Error converting audio", 0.15)
            return None, False
        
        # Get duration of the converted file
        duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                         "-of", "default=noprint_wrappers=1:nokey=1", temp_wav_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        
        try:
            duration = float(duration_result.stdout.strip())
            print(f"ℹ️ Audio duration: {duration:.2f} seconds")
        except (ValueError, IndexError):
            print("⚠️ Could not determine audio duration")
        
        print(f"✅ Audio converted successfully: {temp_wav_path}")
        update_progress("Audio converted successfully", 0.20)
        return temp_wav_path, trimmed
    except Exception as e:
        print(f"❌ Exception during audio conversion: {str(e)}")
        update_progress(f"Error: {str(e)}", 0.15)
        return None, False

def load_pipeline():
    """Load the pyannote pipeline with CLI authentication"""
    update_progress("Loading diarization model", 0.25)
    print("🔄 Loading pyannote speaker diarization model...")
    try:
        # Using latest model version with CLI authentication
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True  # This uses the CLI login token
        )
        
        # Try to move to GPU if available (significantly speeds up processing)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print(f"✅ Using GPU acceleration: {device}")
            pipeline.to(device)
        else:
            print("ℹ️ Running on CPU (no GPU available)")
            
        print("✅ Model loaded successfully!")
        update_progress("Model loaded successfully", 0.30)
        return pipeline
    except Exception as e:
        print(f"❌ Error loading pyannote pipeline: {str(e)}")
        update_progress(f"Pyannote Error: {str(e)}", 0.25)
        return None

def load_whisper_model(model_size="base"):
    """Load the WhisperX model."""
    if not WHISPERX_AVAILABLE:
        print("❌ WhisperX is not installed. Cannot load transcription model.")
        return None

    update_progress(f"Loading WhisperX model ({model_size})", 0.32)
    print(f"🔄 Loading WhisperX model: {model_size}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # WhisperX compute type can be float16 for CUDA, int8 for CPU for faster inference
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        # For WhisperX, model loading is often tied to its transcribe or align methods,
        # but we can preload the base model here.
        # The actual whisperx.load_model is more nuanced, often taking language.
        # We'll keep this simple and let _transcribe_with_whisperx handle specific loading if needed,
        # or assume model_size is enough to "warm up" or select a general model.
        # This function primarily serves to check if a model of this size *can* be conceptually loaded.
        print(f"✅ WhisperX model ({model_size}) ready to be used (actual loading might occur during transcription). Device: {device}, Compute Type: {compute_type}")
        update_progress("WhisperX model ready", 0.34)
        # WhisperX doesn't have a simple "load_model" like standard whisper that returns a generic model object.
        # The model is loaded within its specific functions. So, we return a placeholder or the model_size itself
        # to indicate readiness. The actual model object will be handled by _transcribe_with_whisperx.
        return {"name": model_size, "type": "whisperx", "device": device, "compute_type": compute_type}
    except Exception as e:
        print(f"❌ Error preparing WhisperX model {model_size}: {str(e)}")
        update_progress(f"WhisperX Error: {str(e)}", 0.32)
        return None

def _transcribe_with_whisperx(whisper_model_info, audio_path, options, language, diarization, device):
    """
    Transcribe audio using WhisperX and align with diarization.
    whisper_model_info is a dict like: {"name": model_size, "type": "whisperx", "device": device, "compute_type": compute_type}
    """
    update_progress("Starting WhisperX transcription", 0.45)
    model_name = whisper_model_info.get("name", "base")
    compute_type = whisper_model_info.get("compute_type", "float16" if device == "cuda" else "int8")
    batch_size = options.get("batch_size", 16) # WhisperX default

    try:
        print(f"🔊 Using WhisperX for transcription (model: {model_name}, device: {device}, compute_type: {compute_type})")
        
        # 1. Transcribe
        # The language parameter in whisperx.load_model is for the initial model.
        # If language is None here, WhisperX will detect it.
        loaded_model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        audio = whisperx.load_audio(audio_path)
        result = loaded_model.transcribe(audio, batch_size=batch_size)
        
        # 2. Align whisper output
        if diarization is not None and result.get('segments'):
            print("🔄 Aligning transcript with diarization...")
            # Ensure language is detected for alignment model
            detected_language_for_align = result.get("language", language)
            if not detected_language_for_align:
                 # Attempt to infer language from the first segment if not globally detected
                if result['segments'] and result['segments'][0].get('language'):
                    detected_language_for_align = result['segments'][0]['language']
                else: # Fallback if still no language
                    print("⚠️ Language not detected by WhisperX, defaulting to 'en' for alignment. Results may be suboptimal.")
                    detected_language_for_align = "en"

            model_a, metadata_a = whisperx.load_align_model(language_code=detected_language_for_align, device=device)
            aligned_result = whisperx.align(result["segments"], model_a, metadata_a, audio, device, return_char_alignments=False)
            
            # Assign speaker to segments using diarization
            # This is a simplified version; true alignment might involve whisperx.assign_word_speakers
            # For now, we use the aligned segments and later combine with diarization turns.
            # WhisperX's assign_word_speakers takes diarization directly.
            diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['turn', '_', 'speaker'])
            diarize_df['start'] = diarize_df['turn'].apply(lambda x: x.start)
            diarize_df['end'] = diarize_df['turn'].apply(lambda x: x.end)
            
            # Use assign_word_speakers for speaker assignment
            aligned_result_with_speakers = whisperx.assign_word_speakers(diarize_df, aligned_result)
            result['segments'] = aligned_result_with_speakers["segments"] # Update segments with speaker info
            print("✅ Transcript aligned with diarization.")
        elif not result.get('segments'):
            print("ℹ️ No segments found by WhisperX for alignment.")
        else:
            print("ℹ️ Diarization data not provided or no segments, skipping transcript alignment step with diarization.")

        update_progress("WhisperX transcription complete", 0.80)
        return result
    except Exception as e:
        print(f"❌ Error during WhisperX transcription: {str(e)}")
        update_progress(f"WhisperX Error: {str(e)}", 0.45)
        return None


def transcribe_audio(whisper_model_info, audio_path, language=None, diarization=None):
    """Transcribe audio using WhisperX with speaker diarization"""
    if not WHISPERX_AVAILABLE or whisper_model_info is None:
        print("Transcription skipped: WhisperX not available or model not loaded.")
        return None
        
    try:
        print(f"🔊 Transcribing audio: {audio_path}")
        
        options = {} # WhisperX options can be passed here if needed, e.g., batch_size
        if language and language.strip():
            language = language.strip().lower()
            # options["language"] = language # WhisperX load_model takes language
            
        device = whisper_model_info.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        result = _transcribe_with_whisperx(whisper_model_info, audio_path, options, language, diarization, device)
        
        print("✅ WhisperX transcription processing complete.")
        return result
    except Exception as e:
        print(f"❌ Error transcribing audio with WhisperX: {str(e)}")
        return None

def process_diarization(pipeline, wav_file, known_speakers=None):
    """Process the diarization with progress updates"""
    try:
        update_progress("Starting diarization", 0.35)
        # Build diarization kwargs
        diarization_kwargs = {}
        if known_speakers:
            diarization_kwargs["num_speakers"] = int(known_speakers)
            
        # Process the audio file
        update_progress("Running diarization (this may take a while)", 0.40)
        diarization = pipeline(wav_file, **diarization_kwargs)
        update_progress("Processing diarization results", 0.80)
        
        # Format the results
        output_lines = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = str(timedelta(seconds=round(turn.start)))
            end = str(timedelta(seconds=round(turn.end)))
            label = f"{speaker}"
            output_lines.append(f"{start} - {end} {label}".strip())
            
        # Calculate speaker statistics
        unique_speakers = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
        speaker_times = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if speaker in speaker_times:
                speaker_times[speaker] += duration
            else:
                speaker_times[speaker] = duration
                
        update_progress("Diarization complete", 0.90)
        return diarization, output_lines, unique_speakers, speaker_times
    except Exception as e:
        print(f"❌ Error during diarization: {str(e)}")
        update_progress(f"Error: {str(e)}", 0.35)
        return None, None, None, None

def _build_final_transcript_string(diarization_obj, 
                                   unique_speakers, speaker_times, was_trimmed, trim_start, trim_end,
                                   raw_whisper_result, # From transcribe_audio(), used by WhisperX
                                   is_whisperx_available=True): # Assume WhisperX is always used now
    header = []
    header.append(f"📊 Detected {len(unique_speakers)} speaker(s)")
    for speaker_id, total_time in sorted(speaker_times.items(), key=lambda item: item[0]):
        percentage = (total_time / sum(speaker_times.values()) * 100) if sum(speaker_times.values()) > 0 else 0
        header.append(f"  - {speaker_id}: {total_time:.1f}s ({percentage:.1f}%)")
    
    if was_trimmed:
        header.append(f"Audio trimmed from {timedelta(seconds=round(trim_start))} to {timedelta(seconds=round(trim_end))}")
    header.append("") # Blank line for spacing

    output_lines = []
    if raw_whisper_result and raw_whisper_result.get('segments') and is_whisperx_available:
        print("📝 Formatting output using WhisperX segments with speaker info.")
        # WhisperX result with assign_word_speakers should have 'speaker' in segments
        for segment in raw_whisper_result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment.get('text', '').strip()
            speaker_label = segment.get('speaker', 'UNKNOWN_SPEAKER') 

            start_td = str(timedelta(seconds=round(start_time)))
            end_td = str(timedelta(seconds=round(end_time)))
            output_lines.append(f"{start_td} - {end_td} {speaker_label}: {text}")
    elif diarization_obj: # Fallback to diarization only if transcription failed or no segments
        print("📝 Transcription result not available or no segments, formatting diarization output only.")
        for turn, _, speaker in diarization_obj.itertracks(yield_label=True):
            start = str(timedelta(seconds=round(turn.start)))
            end = str(timedelta(seconds=round(turn.end)))
            output_lines.append(f"{start} - {end} {speaker}")
            
    return "\n".join(header + output_lines)

def _diarize_initial_checks_and_setup(audio_file_path, output_path_param, temp_dir_param):
    """
    Performs initial checks for audio file, Hugging Face login, pipeline loading,
    and sets up effective temporary and output paths.
    Returns: (pipeline_obj, effective_output_path, effective_temp_dir, error_message_str_or_None)
    """
    if audio_file_path is None:
        logging.warning("No audio file provided for diarization.")
        return None, None, None, "⚠️ Please upload an audio file."

    effective_output_path = output_path_param if output_path_param else DEFAULT_OUTPUT_DIR
    effective_temp_dir = temp_dir_param # temp_dir_param already defaults to DEFAULT_TEMP_DIR in `diarize` signature
    
    try:
        os.makedirs(effective_temp_dir, exist_ok=True)
        os.makedirs(effective_output_path, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating directories: {e}", exc_info=True)
        return None, None, None, f"❌ Error creating directories: {e}"


    if not check_hf_login(): # Logs its own error and prints instructions
        return None, None, None, "❌ Please login with huggingface-cli login first. See console for instructions."

    pipeline_obj = load_pipeline() # Logs its own error and prints instructions
    if pipeline_obj is None:
        return None, None, None, "❌ Failed to load the diarization model. See console for details."
    
    return pipeline_obj, effective_output_path, effective_temp_dir, None # Success

def _diarize_transcription_step(
    transcribe_flag, whisper_model_size_param, wav_file_path, 
    language_code, diarization_obj, original_audio_file_path_for_cleanup):
    """Helper function to handle the transcription part of the diarization process, using WhisperX."""
    raw_whisper_result = None

    if transcribe_flag:
        if not WHISPERX_AVAILABLE:
            print("⚠️ Transcription requested but WhisperX is not available. Skipping.")
            update_progress("WhisperX not available, skipping transcription", 0.40)
            return None # No result if WhisperX isn't there

        whisper_model_info = load_whisper_model(whisper_model_size_param)
        if whisper_model_info is None:
            print("❌ Failed to load WhisperX model. Skipping transcription.")
            update_progress("Failed to load WhisperX model", 0.40)
            # Clean up WAV if it was a temporary conversion
            if wav_file_path and wav_file_path != original_audio_file_path_for_cleanup and os.path.exists(wav_file_path):
                try:
                    os.remove(wav_file_path)
                    print(f"🗑️ Cleaned up temporary WAV file: {wav_file_path}")
                except Exception as e:
                    print(f"⚠️ Error cleaning up temp WAV {wav_file_path}: {e}")
            return None # No result if model load fails
        
        # Pass diarization object to transcribe_audio for WhisperX alignment
        raw_whisper_result = transcribe_audio(whisper_model_info, wav_file_path, language_code, diarization_obj)
        
        if raw_whisper_result is None:
            print("⚠️ Transcription with WhisperX failed or returned no result.")
            update_progress("Transcription failed", 0.75) # Or other appropriate progress
        else:
            print("✅ Transcription with WhisperX successful.")
            update_progress("Transcription complete", 0.80)
    else:
        print("ℹ️ Transcription not requested.")
        update_progress("Transcription not requested", 0.40) # Update progress even if skipped

    return raw_whisper_result


def diarize(audio_file, output_path, known_speakers, 
            temp_dir=DEFAULT_TEMP_DIR, trim_start=None, trim_end=None, 
            transcribe=False, whisper_model_size="base", language=None,
            export_json=False):
    """Perform speaker diarization and WhisperX transcription on an audio file"""
    update_progress("Starting process", 0.01)
    
    # 1. Initial Checks & Setup (Paths, HF Login, Pipeline Load)
    pipeline, effective_output_path, effective_temp_dir, error_message = _diarize_initial_checks_and_setup(
        audio_file, output_path, temp_dir
    )
    if error_message:
        return None, error_message # Return None for output_file, and the error string

    wav_file_to_process = None # Ensure it's defined for finally block
    try:
        # 2. Convert Audio to WAV & Trim
        wav_file_to_process, was_trimmed = convert_audio_to_wav(audio_file, effective_temp_dir, trim_start, trim_end)
        if wav_file_to_process is None:
            return None, "❌ Failed to convert audio file. Check logs."

        # 3. Perform Speaker Diarization
        diarization_obj, base_diarization_lines, unique_speakers, speaker_times = process_diarization(
            pipeline, wav_file_to_process, known_speakers
        )
        if diarization_obj is None:
            return None, "❌ Error during pyannote diarization. Check logs."

        # 4. Perform Transcription (if requested) using WhisperX
        raw_whisper_result = _diarize_transcription_step(
            transcribe, whisper_model_size, wav_file_to_process, 
            language, diarization_obj, audio_file # audio_file is original for cleanup check
        )
        # Note: raw_whisper_result will be None if transcription is skipped or fails

        # 5. Build Final Output String (using WhisperX results if available)
        # The 'is_whisperx_available' flag in _build_final_transcript_string is now effectively always True
        # if transcription was attempted and WHISPERX_AVAILABLE is True globally.
        # The function will use raw_whisper_result if it's not None.
        transcript_text_content = _build_final_transcript_string(
            diarization_obj, unique_speakers, speaker_times, was_trimmed, trim_start, trim_end,
            raw_whisper_result, WHISPERX_AVAILABLE 
        )
        
        # 6. Save Outputs
        output_text_file, save_status_messages = _save_all_outputs(
            effective_output_path, audio_file, transcript_text_content,
            export_json, diarization_obj, raw_whisper_result, # Pass raw_whisper_result for JSON
            was_trimmed, trim_start, trim_end
        )
        
        update_progress("Process complete!", 1.0)
        final_status = f"✅ Processing complete for {Path(audio_file).name}.\n{save_status_messages}"
        return output_text_file, final_status

    except Exception as e:
        print(f"❌❌❌ An unexpected error occurred in diarize function: {str(e)}")
        import traceback
        traceback.print_exc()
        update_progress(f"Critical Error: {str(e)}", 0)
        return None, f"❌ An unexpected error occurred: {str(e)}"
    finally:
        # 7. Cleanup temporary WAV file if it was created and is different from original input
        if wav_file_to_process and wav_file_to_process != audio_file and os.path.exists(wav_file_to_process):
            try:
                os.remove(wav_file_to_process)
                print(f"🗑️ Cleaned up temporary WAV file: {wav_file_to_process}")
            except Exception as e:
                print(f"⚠️ Error cleaning up temp WAV {wav_file_to_process}: {e}")

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="Speaker Diarization with pyannote") as demo:
        gr.Markdown("# 🎙️ Speaker Diarization with pyannote.audio v3.1")
        gr.Markdown("Identify and separate speakers in audio recordings (supports M4A, WAV, MP3, etc.)")

        # Hidden radio to track which mode is active
        processing_mode = gr.Radio(
            choices=["Single File", "Batch Processing"],
            value="Single File",
            visible=False,
            label="Processing Mode"
        )

        with gr.Tabs() as interface_tabs:
            with gr.TabItem("Single File", id="single_tab"):
                with gr.Row():
                    audio_file = gr.Audio(label="Upload or record audio", type="filepath")

                with gr.Tabs():
                    with gr.TabItem("Basic Settings"):
                        with gr.Row():
                            output_path = gr.Textbox(label="Output folder", value=DEFAULT_OUTPUT_DIR, info="Where to save the diarization results")
                            known_speakers = gr.Number(label="Number of speakers", precision=0, info="Leave blank for auto detection")
                        include_speaker_labels = gr.Checkbox(label="Include speaker labels", value=True, info="Show speaker IDs in output")
                        transcribe = gr.Checkbox(
                            label="Transcribe speech", 
                            value=WHISPERX_AVAILABLE, 
                            info="Generate text transcription using WhisperX (recommended)"
                        )
                        export_json = gr.Checkbox(label="Export results as JSON", value=False, info="Save structured data for further processing")
                    
                    with gr.TabItem("Advanced Settings"):
                        with gr.Row():
                            temp_dir = gr.Textbox(label="Temporary folder", value=DEFAULT_TEMP_DIR, info="Where to store temporary files")
                        with gr.Row():
                            trim_start = gr.Number(label="Trim start (seconds)", value=None, info="Start time for audio trimming (optional)")
                            trim_end = gr.Number(label="Trim end (seconds)", value=None, info="End time for audio trimming (optional)")
                        
                        # Whisper settings
                        with gr.Row():
                            whisper_model_size = gr.Radio(
                                ["tiny", "base", "small", "medium", "large-v3"], 
                                label="Whisper Model Size", 
                                value="base",
                                info="Larger models are more accurate but slower"
                            )
                            language = gr.Dropdown(
                                ["", "English", "Chinese", "German", "Spanish", "Russian", "Korean", "French", "Japanese", "Portuguese", "Turkish"],
                                label="Language (optional)",
                                info="Specify language for better transcription accuracy"
                            )
                
                # Update mode when this tab is selected
                interface_tabs.select(
                    lambda: "Single File", 
                    inputs=None, 
                    outputs=processing_mode,
                    trigger_mode="once"
                )
            
            with gr.TabItem("Batch Processing", id="batch_tab"):
                with gr.Row():
                    input_folder = gr.Textbox(label="Input folder with audio files", info="Folder containing multiple audio files to process")
                
                with gr.Row():
                    batch_output_path = gr.Textbox(label="Output folder", value=DEFAULT_OUTPUT_DIR, info="Where to save all diarization results")
                    batch_known_speakers = gr.Number(label="Number of speakers", precision=0, info="Leave blank for auto detection")
                
                batch_include_speaker_labels = gr.Checkbox(label="Include speaker labels", value=True, info="Show speaker IDs in output")
                batch_transcribe = gr.Checkbox(
                    label="Transcribe speech", 
                    value=WHISPERX_AVAILABLE, 
                    info="Generate text transcription using WhisperX (recommended)"
                )
                batch_export_json = gr.Checkbox(label="Export results as JSON", value=False, info="Save structured data for further processing")
                
                with gr.Accordion("Advanced Batch Settings", open=False):
                    batch_temp_dir = gr.Textbox(label="Temporary folder", value=DEFAULT_TEMP_DIR, info="Where to store temporary files")
                    
                    # Whisper settings for batch
                    with gr.Row():
                        batch_whisper_model_size = gr.Radio(
                            ["tiny", "base", "small", "medium", "large-v3"], 
                            label="Whisper Model Size", 
                            value="base",
                            info="Larger models are more accurate but slower"
                        )
                        batch_language = gr.Dropdown(
                            ["", "English", "Chinese", "German", "Spanish", "Russian", "Korean", "French", "Japanese", "Portuguese", "Turkish"],
                            label="Language (optional)",
                            info="Specify language for better transcription accuracy"
                        )
                
                # Update mode when this tab is selected
                interface_tabs.select(
                    lambda: "Batch Processing", 
                    inputs=None, 
                    outputs=processing_mode,
                    trigger_mode="once"
                )

        with gr.Row():
            run_btn = gr.Button("Run Diarization", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="stop")

        ui_progress_bar = gr.Progress() # Renamed from progress
        progress_info = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            output_text = gr.Textbox(label="Diarization Output", lines=20, interactive=False)
            output_file = gr.Textbox(label="Output File Path", interactive=False, visible=False)

        # Download button for output file
        download_btn = gr.Button("Download Results", interactive=False)
        
        def enable_download_button(output_file_path):
            if output_file_path:
                return gr.Button(interactive=True)
            else:
                return gr.Button(interactive=False)
        
        def on_cancel():
            update_progress("Cancelled by user", 0)
            return "⚠️ Operation cancelled by user", None

        # Process either single file or batch depending on active tab
        def process_audio(mode, audio_file_in, output_path_in, include_speaker_labels_in, known_speakers_in, 
                         temp_dir_in, trim_start_in, trim_end_in, transcribe_in, whisper_model_size_in, language_in, export_json_in,
                         input_folder_in, batch_output_path_in, batch_include_speaker_labels_in, batch_known_speakers_in,
                         batch_temp_dir_in, batch_transcribe_in, batch_whisper_model_size_in, batch_language_in, batch_export_json_in):
            
            # Map language display names to Whisper codes
            # Default to None (auto-detect) if the key isn't in the map, or if language_in is already None/empty.
            # The .get method with a default handles cases where language_in might be something unexpected,
            # though the dropdown restricts choices.
            mapped_language_single = LANGUAGE_NAME_TO_CODE_MAP.get(language_in, None)
            mapped_language_batch = LANGUAGE_NAME_TO_CODE_MAP.get(batch_language_in, None)

            if mode == "Single File":  # Single file mode
                # Note: include_speaker_labels_in is not used by diarize function
                return diarize(audio_file_in, output_path_in, known_speakers_in, 
                             temp_dir_in, trim_start_in, trim_end_in, 
                             transcribe_in, whisper_model_size_in, mapped_language_single, export_json_in)
            else:  # Batch mode
                # Note: batch_include_speaker_labels_in is not used by batch_diarize function
                return batch_diarize(input_folder_in, batch_output_path_in, batch_known_speakers_in,
                                    batch_temp_dir_in, batch_transcribe_in, 
                                    batch_whisper_model_size_in, mapped_language_batch, batch_export_json_in)

        # Set up callbacks
        run_btn.click(
            process_audio,
            inputs=[
                processing_mode,
                audio_file, output_path, include_speaker_labels, known_speakers, 
                temp_dir, trim_start, trim_end, transcribe, whisper_model_size, language, export_json,
                input_folder, batch_output_path, batch_include_speaker_labels, batch_known_speakers,
                batch_temp_dir, batch_transcribe, batch_whisper_model_size, batch_language, batch_export_json
            ],
            outputs=[output_text, output_file]
        )
        
        cancel_btn.click(
            on_cancel,
            inputs=[],
            outputs=[output_text, output_file]
        )
        
        output_file.change(
            enable_download_button,
            inputs=[output_file],
            outputs=[download_btn]
        )
        
        # Renamed function and modified signature to capture ui_progress_bar
        def _update_progress_bar_and_status_text(progress_bar_ref=ui_progress_bar):
            progress_bar_ref(0, desc="Starting...") # Use the captured reference
            status = get_progress()
            last_progress = 0
            
            while status["progress"] < 1.0:
                time.sleep(0.1)
                status = get_progress()
                if status["progress"] > last_progress:
                    progress_bar_ref(status["progress"], desc=status["status"]) # Use the captured reference
                    last_progress = status["progress"]
                    
            progress_bar_ref(1.0, desc="Completed") # Use the captured reference
            return status["status"]
        
        run_btn.click(
            _update_progress_bar_and_status_text, # Use the new function name
            inputs=[], # Removed ui_progress_bar from inputs
            outputs=[progress_info]
        )

        # Enable/disable transcription based on WhisperX availability
        def update_transcription_availability(value):
            if value and not WHISPERX_AVAILABLE:
                return gr.Checkbox(value=False, interactive=False, 
                                  info="WhisperX not found. Transcription disabled. Please install following README instructions.")
            return gr.Checkbox(interactive=True)
            
        transcribe.change(
            update_transcription_availability,
            inputs=[transcribe],
            outputs=[transcribe]
        )
        
        batch_transcribe.change(
            update_transcription_availability,
            inputs=[batch_transcribe],
            outputs=[batch_transcribe]
        )

        # Help section with organized documentation
        with gr.Accordion("📚 Help & Documentation", open=False):
            with gr.Tabs():
                with gr.TabItem("Getting Started"):
                    gr.Markdown("""
                    ### 🚀 Quick Start Guide
                    
                    1. **Login to Hugging Face**:
                       ```
                       huggingface-cli login
                       ```
                       Enter your token when prompted (create one at https://huggingface.co/settings/tokens)
                    
                    2. **Accept Model Licenses**:
                       - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
                       - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
                       - Click "Access repository" on both pages
                    
                    3. **Upload Audio**:
                       - Choose an audio file in any common format
                       - Adjust settings as needed
                       - Click "Run Diarization"
                    
                    4. **View Results**:
                       - See speaker segmentation with timestamps
                       - Results are saved to the specified output folder
                    """)
                
                with gr.TabItem("File Formats"):
                    gr.Markdown("""
                    ### 📋 Supported Audio Formats
                    
                    This tool supports all common audio formats through FFmpeg conversion:
                    
                    | Format | Description |
                    |--------|-------------|
                    | WAV    | Waveform Audio File (best quality/compatibility) |
                    | MP3    | MPEG Audio Layer III (common compressed format) |
                    | M4A    | MPEG-4 Audio (Apple format, commonly from recordings) |
                    | FLAC   | Free Lossless Audio Codec (high quality compressed) |
                    | OGG    | Ogg Vorbis (free, open compressed format) |
                    | AAC    | Advanced Audio Coding (common compressed format) |
                    
                    Any other format supported by FFmpeg should also work. Files are automatically 
                    converted to 16kHz mono WAV format for processing.
                    """)
                
                with gr.TabItem("Features"):
                    gr.Markdown("""
                    ### 🔧 Key Features
                    
                    #### Basic Features
                    - **Speaker Diarization**: Identify who spoke when
                    - **Multiple File Formats**: Process WAV, MP3, M4A, FLAC and more
                    - **Speaker Statistics**: See speaking time and percentages for each speaker
                    
                    #### Advanced Features
                    - **Batch Processing**: Process multiple audio files at once
                    - **Audio Trimming**: Process only a specific part of an audio file
                    - **Transcription**: Convert speech to text using OpenAI Whisper
                    - **Structured Output**: Export results as JSON for further processing
                    - **GPU Acceleration**: Automatically uses GPU if available
                    - **Progress Tracking**: See real-time progress for long processing tasks
                    """)
                
                with gr.TabItem("Transcription"):
                    gr.Markdown(f"""
                    ### 🎯 Speech Transcription 
                    
                    **Current Status**: 
                    - WhisperX: {"Available ✅" if WHISPERX_AVAILABLE else "Not Available ❌"} (Recommended)
                    
                    #### Installation
                    If transcription is not available, install WhisperX (recommended):
                    ```
                    pip install git+https://github.com/m-bain/whisperx.git
                    ```
                    
                    #### WhisperX Advantages
                    - Better integration with speaker diarization
                    - More accurate word-level timestamps
                    - Voice activity detection (VAD) for better noise handling
                    - Faster processing than standard Whisper
                    
                    #### Model Sizes
                    | Size    | Memory | Accuracy | Speed | Use Case |
                    |---------|--------|----------|-------|----------|
                    | tiny    | ~1GB   | Low      | Fast  | Quick testing |
                    | base    | ~1GB   | Basic    | Fast  | General use |
                    | small   | ~2GB   | Good     | Medium| Better accuracy |
                    | medium  | ~5GB   | High     | Slow  | High accuracy |
                    | large-v3| ~10GB  | Highest  | Slowest| Best results |
                    
                    Specifying a language can improve transcription accuracy significantly.
                    """)
                
                with gr.TabItem("JSON Export"):
                    gr.Markdown("""
                    ### 📊 Structured JSON Output
                    
                    When enabling "Export results as JSON", you get a structured file with:
                    
                    ```json
                    {
                      "file_info": {
                        "filename": "recording.mp3",
                        "path": "/path/to/recording.mp3",
                        "trimmed": false
                      },
                      "segments": [
                        {
                          "start": 0.5,
                          "end": 10.2,
                          "duration": 9.7,
                          "speaker": "SPEAKER_01",
                          "text": "This is the transcribed text if available"
                        },
                        ...
                      ],
                      "speakers": {
                        "SPEAKER_01": {
                          "talk_time": 120.5,
                          "percentage": 45.3,
                          "segments": [0, 2, 5, ...]
                        },
                        ...
                      }
                    }
                    ```
                    
                    This format is ideal for:
                    - Further data analysis
                    - Integration with other tools
                    - Building custom visualizations
                    - Training machine learning models
                    """)
                
                with gr.TabItem("Troubleshooting"):
                    gr.Markdown("""
                    ### ⚠️ Common Issues & Solutions
                    
                    #### Authentication Issues
                    - **Error**: "Not logged in to Hugging Face CLI"
                      **Solution**: Run `huggingface-cli login` and enter your token
                    
                    - **Error**: "401 Unauthorized"
                      **Solution**: Make sure you've accepted the model licenses on the Hugging Face website
                    
                    #### Audio Conversion Problems
                    - **Error**: "Failed to convert audio file"
                      **Solution**: Install FFmpeg or check if your audio file is corrupted
                    
                    #### Model Loading Failures
                    - **Error**: "ONNX Runtime Error"
                      **Solution**: Try `pip uninstall onnxruntime onnxruntime-gpu` then `pip install onnxruntime==1.15.1`
                    
                    - **Error**: "SSL Error"
                      **Solution**: Run `pip install --upgrade certifi`
                    
                    #### Transcription Problems
                    - **Error**: "Failed to load Whisper model"
                      **Solution**: Check if you have enough memory for the selected model size, try a smaller model
                    
                    For further help, check the console output for detailed error messages.
                    """)
                
                with gr.TabItem("Requirements"):
                    gr.Markdown("""
                    ### 📦 System Requirements
                    
                    #### Essential Dependencies
                    - **Python**: 3.8 or newer
                    - **PyTorch**: 1.12.0 or newer
                    - **FFmpeg**: For audio conversion
                    - **pyannote.audio**: For diarization
                    - **Hugging Face CLI**: For authentication
                    
                    #### Optional Dependencies
                    - **OpenAI Whisper**: For speech transcription
                    - **CUDA**: For GPU acceleration (significantly speeds up processing)
                    
                    #### Hardware Recommendations
                    - **CPU**: Modern multi-core processor
                    - **RAM**: 8GB minimum, 16GB+ recommended
                    - **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster processing)
                    - **Disk Space**: At least 10GB free space for models and temporary files
                    
                    #### Internet Connection
                    Required for initial model download and authentication.
                    """)

        # Bottom copyright/version info 
        gr.Markdown("""
        ---
        *Speaker Diarization Tool v3.1 | Powered by pyannote.audio and Whisper | © 2025*
        """)
        
    return demo

if __name__ == "__main__":
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is installed and working")
    except Exception:
        print("❌ FFmpeg is not installed or not in PATH. Please install FFmpeg to process audio files.")
    
    # Check transcription availability
    if WHISPERX_AVAILABLE:
        print("✅ WhisperX is installed (recommended for better transcription)")
    else:
        print("❌ WhisperX is not installed. This tool requires WhisperX for transcription.")
        print("   Please install it by ensuring it's in your requirements.txt and running:")
        print("   pip install -r requirements.txt")
    
    # Print version info
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    try:
        import pyannote.audio
        print(f"🎙️ pyannote.audio version: {pyannote.audio.__version__}")
    except (ImportError, AttributeError):
        print("⚠️ Could not determine pyannote.audio version")
    
    # Create default directories
    os.makedirs(DEFAULT_TEMP_DIR, exist_ok=True)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    # Check login status on startup
    update_progress("Starting application", 0.0)
    check_hf_login()
    print("🚀 Launching Gradio interface...")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch() 
