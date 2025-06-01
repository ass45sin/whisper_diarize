#!/usr/bin/env python3
"""
Speaker diarization using pyannote with HuggingFace CLI authentication
This script uses CLI authentication which is more reliable than token-based auth
Make sure to run 'huggingface-cli login' before running this script
Uses pyannote/speaker-diarization-3.1 for improved performance
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

# Check for whisper/whisperx availability
WHISPERX_AVAILABLE = False
WHISPER_AVAILABLE = False
WHISPERX_VERSION = None

try:
    import whisperx
    import importlib.metadata
    try:
        WHISPERX_VERSION = importlib.metadata.version("whisperx")
        print(f"✅ WhisperX is available (recommended), version: {WHISPERX_VERSION}")
    except importlib.metadata.PackageNotFoundError:
        WHISPERX_VERSION = "unknown"
        print("✅ WhisperX is available (recommended), version: unknown")
    WHISPERX_AVAILABLE = True
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        print("✅ Standard Whisper is available")
    except ImportError:
        print("⚠️ Neither WhisperX nor Whisper is available")

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
        error_msg = str(e)
        print(f"❌ Error loading model: {error_msg}")
        update_progress(f"Error loading model: {error_msg[:50]}...", 0.25)
        
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n===================================================================")
            print("AUTHORIZATION ERROR:")
            print("1. Make sure you've accepted the license agreements at:")
            print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("   - https://huggingface.co/pyannote/segmentation-3.0")
            print("2. Make sure you're logged in with the same account that accepted the licenses")
            print("3. Try logging in again with: huggingface-cli login")
            print("===================================================================\n")
        elif "SSL" in error_msg:
            print("\n===================================================================")
            print("SSL ERROR: There might be a network or proxy issue")
            print("Try running: pip install --upgrade certifi")
            print("===================================================================\n")
        elif "onnxruntime" in error_msg.lower():
            print("\n===================================================================")
            print("ONNX RUNTIME ERROR: This is a known issue with the model")
            print("Try running: pip uninstall onnxruntime onnxruntime-gpu")
            print("Then: pip install onnxruntime==1.15.1")
            print("===================================================================\n")
        elif "not found" in error_msg.lower() or "could not download" in error_msg.lower():
            print("\n===================================================================")
            print("MODEL NOT FOUND ERROR: The model file couldn't be downloaded")
            print("1. Check your internet connection")
            print("2. Make sure you have accepted all model licenses")
            print("3. Try clearing the cache: rm -rf ~/.cache/huggingface")
            print("===================================================================\n")
        else:
            print("\n===================================================================")
            print("UNEXPECTED ERROR: Please check your installation")
            print("1. Try: pip install --upgrade pyannote.audio")
            print("2. Ensure all dependencies are installed")
            print("3. Try restarting your environment")
            print("===================================================================\n")
        return None

def _try_load_whisperx(model_size):
    try:
        update_progress("Loading WhisperX model", 0.32)
        print(f"🔄 Loading WhisperX {model_size} model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
        print(f"✅ WhisperX model loaded successfully on {device} using {compute_type}!")
        update_progress("WhisperX model loaded", 0.35)
        return model
    except Exception as e:
        print(f"❌ Error loading WhisperX model: {str(e)}")
        return None

def _try_load_standard_whisper(model_size, is_fallback=False):
    progress_msg = "Loading Whisper model as fallback" if is_fallback else "Loading Whisper model"
    try:
        update_progress(progress_msg, 0.32)
        print(f"🔄 Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        print("✅ Whisper model loaded successfully!")
        update_progress("Whisper model loaded", 0.35)
        return model
    except Exception as e:
        error_msg = f"❌ Error loading fallback Whisper model: {str(e)}" if is_fallback else f"❌ Error loading Whisper model: {str(e)}"
        print(error_msg)
        return None

def load_whisper_model(model_size="base"):
    """Load Whisper/WhisperX model for transcription"""
    if WHISPERX_AVAILABLE:
        model = _try_load_whisperx(model_size)
        if model:
            return model
        # Fallback if WhisperX failed
        print("⚠️ Falling back to standard Whisper due to WhisperX error")
        if WHISPER_AVAILABLE:
            return _try_load_standard_whisper(model_size, is_fallback=True)
        return None # WhisperX failed, and standard Whisper not available
    elif WHISPER_AVAILABLE:
        return _try_load_standard_whisper(model_size)
    
    # Neither WhisperX nor standard Whisper is available
    return None

def _transcribe_with_whisperx(whisper_model, audio_path, options, language, diarization, device):
    print("🔊 Using WhisperX for transcription")
    # Initial transcription
    transcription_result = whisper_model.transcribe(audio_path, **options)
    
    try:
        # Alignment and speaker assignment
        detected_lang = language if language else transcription_result.get("language", "en")
        print(f"Using language '{detected_lang}' for alignment")
        
        try:
            model_a, metadata = whisperx.load_align_model(language=detected_lang, device=device)
        except TypeError:
            print(f"Falling back to alternative alignment API (language={detected_lang})")
            model_a, metadata = whisperx.load_align_model(detected_lang, device)
        
        # Align the results
        alignment_output = whisperx.align(
            transcription_result["segments"], # Pass the list of segments from initial transcription
            model_a, 
            metadata, 
            audio_path, 
            device, 
            return_char_alignments=False
        )
        
        # If we have diarization data, integrate speaker information
        if diarization is not None:
            update_progress("Assigning speakers to transcription", 0.45)
            print("🔊 Assigning speakers to transcription segments")
            
            pyannote_speaker_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                pyannote_speaker_turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker # This is pyannote's label (e.g., SPEAKER_00)
                })
            
            if pyannote_speaker_turns: # Ensure there are turns to process
                diarization_df = pd.DataFrame(pyannote_speaker_turns)
                
                # whisperx.assign_word_speakers takes the diarization DataFrame 
                # and the *entire output dictionary from whisperx.align*.
                # Its output, result_after_assignment, is also a dictionary.
                result_after_assignment = whisperx.assign_word_speakers(diarization_df, alignment_output)
                
                # The actual list of segments (with speaker info) is under the "segments" key.
                final_processed_segments_list = result_after_assignment["segments"]
            else: # No pyannote turns to process
                # Use the segments directly from the alignment output
                final_processed_segments_list = alignment_output["segments"]
        else: # No diarization object provided
            # Use the segments directly from the alignment output
            final_processed_segments_list = alignment_output["segments"]
            
        print("✅ WhisperX transcription complete (with alignment/speaker assignment attempt)")
        # Return a structure where "segments" holds the LIST of segment dictionaries
        return {"segments": final_processed_segments_list, "language": detected_lang} 
        
    except Exception as align_error:
        print(f"❌ Warning: Could not align or assign speakers: {str(align_error)}")
        print("⚠️ Continuing with basic WhisperX transcription without alignment")
        print("✅ WhisperX transcription complete (basic)")
        return transcription_result # Return basic transcription if alignment/assignment fails

def transcribe_audio(whisper_model, audio_path, language=None, diarization=None):
    """Transcribe audio using Whisper or WhisperX with speaker diarization"""
    if whisper_model is None:
        return None
        
    try:
        update_progress("Transcribing audio", 0.37)
        print(f"🔊 Transcribing audio: {audio_path}")
        
        options = {}
        if language and language.strip():
            language = language.strip().lower()
            options["language"] = language
            
        if WHISPERX_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            result = _transcribe_with_whisperx(whisper_model, audio_path, options, language, diarization, device)
        else:
            # Fallback to standard Whisper
            print("🔊 Using standard Whisper for transcription")
            result = whisper_model.transcribe(audio_path, **options)
            print("✅ Whisper transcription complete")
        
        update_progress("Transcription complete", 0.50)
        return result
    except Exception as e:
        print(f"❌ Error transcribing audio: {str(e)}")
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

def _get_overlapping_text(turn_start, turn_end, transcript_map):
    overlapping_texts = []
    for (t_start, t_end), text in transcript_map.items():
        # Check for overlap:
        # A_start <= B_end AND A_end >= B_start
        if t_start <= turn_end and t_end >= turn_start:
            overlapping_texts.append(text)
    return " ".join(overlapping_texts).strip()

def combine_diarization_with_transcript(diarization, transcript_result):
    """Combine diarization results with transcript from Whisper"""
    if transcript_result is None or diarization is None:
        return None
    
    try:
        # Extract segments from Whisper
        segments = transcript_result.get("segments", [])
        
        # Create a map of time segments to text
        transcript_map = {}
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            transcript_map[(start_time, end_time)] = text
        
        # Match diarization with transcripts
        result_lines = [] # Renamed from 'result' to 'result_lines' for clarity
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            
            combined_text = _get_overlapping_text(start_time, end_time, transcript_map)
            
            # Format the output
            start_str = str(timedelta(seconds=round(start_time)))
            end_str = str(timedelta(seconds=round(end_time)))
            
            if combined_text:
                result_lines.append(f"{start_str} - {end_str} {speaker}: {combined_text}")
            else:
                result_lines.append(f"{start_str} - {end_str} {speaker}")
        
        return result_lines
    except Exception as e:
        print(f"❌ Error combining diarization and transcript: {str(e)}")
        return None

def _populate_initial_json_from_diarization(diarization, results):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_data = {
            "start": turn.start, "end": turn.end,
            "duration": turn.end - turn.start,
            "speaker": speaker, "text": None
        }
        results["segments"].append(segment_data)
        
        speaker_stats = results["speakers"].setdefault(speaker, {"talk_time": 0, "segments": []})
        speaker_stats["talk_time"] += segment_data["duration"]
        speaker_stats["segments"].append(len(results["segments"]) - 1)

def _add_transcripts_to_json_segments(json_segments, transcript_result):
    whisper_segments_list = transcript_result.get("segments", [])
    
    for i, target_segment in enumerate(json_segments):
        segment_start = target_segment["start"]
        segment_end = target_segment["end"]
        
        matching_texts = []
        for ws in whisper_segments_list:
            w_start, w_end = ws["start"], ws["end"]
            if w_start <= segment_end and w_end >= segment_start:
                matching_texts.append(ws["text"])
        
        if matching_texts:
            json_segments[i]["text"] = " ".join(matching_texts).strip()

def _calculate_speaker_percentages(speakers_data):
    total_talk_time = sum(data["talk_time"] for data in speakers_data.values())
    if total_talk_time == 0:
        for speaker_id in speakers_data:
            speakers_data[speaker_id]["percentage"] = 0
        return

    for speaker_id in speakers_data:
        speakers_data[speaker_id]["percentage"] = \
            (speakers_data[speaker_id]["talk_time"] / total_talk_time * 100)

def format_results_as_json(diarization, transcript_result=None, file_info=None):
    """Format diarization results as structured JSON for further processing"""
    if diarization is None:
        return None
    
    results = {
        "file_info": file_info or {},
        "segments": [],
        "speakers": {}
    }
    
    if file_info and "path" in file_info:
        results["file_info"]["filename"] = os.path.basename(file_info["path"])
        
    _populate_initial_json_from_diarization(diarization, results)
    
    if transcript_result:
        # Pass results["segments"] to be modified in place
        _add_transcripts_to_json_segments(results["segments"], transcript_result)
        
    # Pass results["speakers"] to be modified in place
    _calculate_speaker_percentages(results["speakers"])
    
    return results

def save_json_output(results, output_path, file_name):
    """Save diarization results as JSON file"""
    if results is None:
        return None
    
    try:
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Create the output file name
        json_file = os.path.join(output_path, f"{file_name}_diarization.json")
        
        # Save the results to a JSON file
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return json_file
    except Exception as e:
        print(f"❌ Error saving JSON: {str(e)}")
        return None

def _build_final_transcript_string(diarization_obj, 
                                   unique_speakers, speaker_times, was_trimmed, trim_start, trim_end, # For header
                                   raw_whisper_result, # From transcribe_audio(), used by WhisperX path
                                   combined_std_whisper_lines, # From combine_diarization_with_transcript(), used by std Whisper path
                                   is_whisperx_available):
    header = []
    header.append(f"📊 Detected {len(unique_speakers)} speaker(s)")
    
    total_speaker_time = sum(speaker_times.values())
    
    sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
    for sp, dur in sorted_speakers:
        perc = (dur / total_speaker_time * 100) if total_speaker_time > 0 else 0
        header.append(f"  - {sp}: {dur:.1f}s ({perc:.1f}%)")
    
    if was_trimmed:
        header.append(f"✂️ Audio was trimmed to {trim_start}s - {trim_end}s")

    output_lines = []

    if is_whisperx_available and raw_whisper_result and raw_whisper_result.get("segments"):
        # WhisperX path: Iterate directly through WhisperX's own speaker-assigned segments.
        # These segments should already have speaker labels if assign_words_to_speakers worked.
        for segment in raw_whisper_result.get("segments", []):
            start_time = segment.get("start")
            end_time = segment.get("end")
            text_content = segment.get("text", "").strip()
            # Use speaker from WhisperX segment, default if not present
            speaker_label = segment.get("speaker") # Get speaker, could be None or empty

            if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                start_str = str(timedelta(seconds=round(start_time)))
                end_str = str(timedelta(seconds=round(end_time)))
                if speaker_label: # If speaker_label is not None and not an empty string
                    output_lines.append(f"{start_str} - {end_str} {speaker_label}: {text_content}")
                else: # No speaker label from WhisperX, or it's empty
                    output_lines.append(f"{start_str} - {end_str}: {text_content}")
            else:
                # Fallback if timing info is missing/invalid in a segment from WhisperX
                # Preserve speaker label if available, otherwise print generic timing error
                if speaker_label:
                    output_lines.append(f"TIMING_ERROR {speaker_label}: {text_content}")
                else:
                    output_lines.append(f"TIMING_ERROR: {text_content}")
    elif combined_std_whisper_lines: 
        # Standard Whisper: combined_std_whisper_lines are already pyannote-turn-aligned.
        output_lines.extend(combined_std_whisper_lines)
    else: 
        # No transcription, or WhisperX result was not usable in the expected way.
        # Fallback to listing pyannote turns without text.
        if diarization_obj:
            for turn, _, speaker_label_from_pyannote in diarization_obj.itertracks(yield_label=True):
                start_str = str(timedelta(seconds=round(turn.start)))
                end_str = str(timedelta(seconds=round(turn.end)))
                output_lines.append(f"{start_str} - {end_str} {speaker_label_from_pyannote}")
            
    return "\n".join(header) + "\n\n" + "\n".join(output_lines)

def _save_all_outputs(output_path_dir, original_audio_filepath, transcript_content_to_save,
                      export_json_flag, diarization_obj, raw_whisper_result_for_json,
                      was_trimmed, trim_start, trim_end):
    messages = []
    text_file_path = None
    base_name = os.path.splitext(os.path.basename(original_audio_filepath))[0]

    try:
        update_progress("Saving results", 0.95)
        text_file_path = os.path.join(output_path_dir, f"{base_name}_diarization.txt")
        with open(text_file_path, "w") as f:
            f.write(transcript_content_to_save)
        messages.append(f"\n\n✅ Saved to: {text_file_path}") # Start with double newline if it's the first message
        
        if export_json_flag:
            file_info = {
                "path": original_audio_filepath, "trimmed": was_trimmed,
                "trim_start": trim_start if was_trimmed else None,
                "trim_end": trim_end if was_trimmed else None
            }
            json_data = format_results_as_json(diarization_obj, raw_whisper_result_for_json, file_info)
            json_file = save_json_output(json_data, output_path_dir, base_name) # save_json_output returns path or None
            if json_file:
                messages.append(f"\n✅ JSON exported to: {json_file}")
            # No explicit message if json_file is None, as save_json_output prints its own error
                
    except Exception as e:
        # This message will appear if the .txt save fails, or if saving setup (like basename) fails.
        messages.append(f"\n\n⚠️ Failed to save to file(s): {str(e)}")

    return text_file_path, "".join(messages)

# --- Helper Functions for `diarize` ---

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
    """
    Handles the transcription part of the diarization process.
    Returns: (raw_whisper_result, combined_std_whisper_lines, error_message_str_or_None)
    """
    if not transcribe_flag or not (WHISPERX_AVAILABLE or WHISPER_AVAILABLE):
        return None, None, None # No transcription requested or no Whisper available

    whisper_model = load_whisper_model(whisper_model_size_param)
    if whisper_model is None:
        # Error message already logged by load_whisper_model.
        # Cleanup the temporary WAV file if it was created for this transcription attempt.
        if wav_file_path and wav_file_path != original_audio_file_path_for_cleanup:
            try: os.remove(wav_file_path)
            except OSError as e: logging.warning(f"Could not remove temp WAV {wav_file_path} during transcription model load failure: {e}")
        return None, None, "❌ Failed to load transcription model. See console for details."
    
    diar_for_transcribe = diarization_obj if WHISPERX_AVAILABLE else None
    raw_whisper_result = transcribe_audio(whisper_model, wav_file_path, language_code, diar_for_transcribe)
    
    combined_std_whisper_lines = None
    if raw_whisper_result and not WHISPERX_AVAILABLE: # Standard Whisper path needs manual combination
        combined_std_whisper_lines = combine_diarization_with_transcript(diarization_obj, raw_whisper_result)
        if combined_std_whisper_lines is None:
            logging.warning("Failed to combine standard Whisper transcript with diarization for single file processing.")
            # Not returning an error message here, as some transcript (raw_whisper_result) might still be usable.

    return raw_whisper_result, combined_std_whisper_lines, None # Success


def diarize(audio_file, output_path, known_speakers, 
            temp_dir=DEFAULT_TEMP_DIR, trim_start=None, trim_end=None, 
            transcribe=False, whisper_model_size="base", language=None,
            export_json=False):
    """Perform speaker diarization on an audio file"""
    update_progress("Starting", 0.0)
    start_time = time.time()
    
    if audio_file is None:
        return "⚠️ Please upload an audio file.", None

    # Ensure output_path is set (it's used by _save_all_outputs)
    # The original code for output_path creation:
    # if output_path:
    #     os.makedirs(output_path, exist_ok=True)
    # else:
    #     output_path = DEFAULT_OUTPUT_DIR
    #     os.makedirs(output_path, exist_ok=True)
    # This logic needs to be before _save_all_outputs might be called,
    # but _save_all_outputs is only called if this output_path is valid.
    # Let's ensure it's created early.
    current_output_path = output_path if output_path else DEFAULT_OUTPUT_DIR
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(current_output_path, exist_ok=True) # Use current_output_path here

    if not check_hf_login():
        return "❌ Please login with huggingface-cli login first", None

    pipeline = load_pipeline()
    if pipeline is None:
        return "❌ Failed to load the diarization model. See console for details.", None

    wav_file, was_trimmed = convert_audio_to_wav(audio_file, temp_dir, trim_start, trim_end)
    if wav_file is None:
        return "❌ Failed to convert audio file. Check if the format is supported by ffmpeg.", None

    print(f"🔊 Diarizing {wav_file}...")
    diarization, base_diarization_lines, unique_speakers, speaker_times = process_diarization(
        pipeline, wav_file, known_speakers
    )
    
    if diarization is None:
        if wav_file and wav_file != audio_file: os.remove(wav_file)
        return "❌ Diarization failed. See console for details.", None

    # Transcription
    raw_whisper_result = None
    combined_std_whisper_lines = None
    if transcribe and (WHISPERX_AVAILABLE or WHISPER_AVAILABLE):
        whisper_model = load_whisper_model(whisper_model_size)
        if whisper_model is None:
            if wav_file and wav_file != audio_file: os.remove(wav_file)
            return "❌ Failed to load transcription model. See console for details.", None
        
        diar_for_transcribe = diarization if WHISPERX_AVAILABLE else None
        raw_whisper_result = transcribe_audio(whisper_model, wav_file, language, diar_for_transcribe)
        
        if raw_whisper_result and not WHISPERX_AVAILABLE:
            combined_std_whisper_lines = combine_diarization_with_transcript(diarization, raw_whisper_result)

    # Cleanup temporary wav file
    if wav_file and wav_file != audio_file:
        try: os.remove(wav_file)
        except: pass

    # Build the main text content
    # base_diarization_lines comes from process_diarization
    transcript_text_content = _build_final_transcript_string(
        diarization, # Pass the full diarization object
        unique_speakers, speaker_times, was_trimmed, trim_start, trim_end,
        raw_whisper_result, combined_std_whisper_lines, WHISPERX_AVAILABLE
    )

    # Save files and get status messages
    # current_output_path is the actual directory where files will be saved.
    # The original code used output_path directly in the saving block, which could be None initially.
    # My _save_all_outputs helper expects a valid directory.
    saved_text_file_actual_path = None
    save_status_messages = ""

    # The original code had `if output_path:` to decide if saving happens.
    # Here, `current_output_path` will always be a valid path (either user-provided or default).
    # So, saving will always be attempted. The `output_file` return for Gradio relies on this.
    # The user might provide an empty string for output_path in Gradio UI, which then defaults.
    # The Gradio interface has `output_path = gr.Textbox(label="Output folder", value=DEFAULT_OUTPUT_DIR, ...)`
    # So `output_path` from Gradio will be a string. If user clears it, it might be empty string.
    # If `output_path` is an empty string, `current_output_path` becomes `DEFAULT_OUTPUT_DIR`.
    # So, saving should always happen to `current_output_path`.

    saved_text_file_actual_path, save_status_messages = _save_all_outputs(
        current_output_path, audio_file, transcript_text_content,
        export_json, diarization, raw_whisper_result,
        was_trimmed, trim_start, trim_end
    )
    
    final_display_text = transcript_text_content + save_status_messages
    
    elapsed_time = time.time() - start_time
    final_display_text += f"\n\n⏱️ Processing time: {elapsed_time:.2f} seconds"
    
    update_progress("Completed", 1.0)
    return final_display_text, saved_text_file_actual_path

def batch_diarize(input_folder, output_path, known_speakers, 
                 temp_dir=DEFAULT_TEMP_DIR, transcribe=False, whisper_model_size="base", language=None,
                 export_json=False):
    """Process multiple audio files in a folder"""
    update_progress("Starting batch processing", 0.0)
    
    if not os.path.isdir(input_folder):
        return "⚠️ Input folder does not exist or is not a directory.", None
    
    # Find all audio files in the input folder
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
    
    if not audio_files:
        return f"⚠️ No audio files found in {input_folder} with extensions {', '.join(audio_extensions)}", None
    
    # Create output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Check login once for all files
    if not check_hf_login():
        return "❌ Please login with huggingface-cli login first", None
    
    # Load models once for all files
    pipeline = load_pipeline()
    if pipeline is None:
        return "❌ Failed to load diarization model", None
    
    whisper_model = None
    if transcribe and (WHISPERX_AVAILABLE or WHISPER_AVAILABLE):
        whisper_model = load_whisper_model(whisper_model_size)
        if whisper_model is None and transcribe:
            return "❌ Failed to load transcription model", None
    
    # Process each file
    results = []
    total_files = len(audio_files)
    
    for i, audio_file in enumerate(audio_files):
        file_name = os.path.basename(audio_file)
        file_progress = i / total_files
        overall_progress = 0.1 + (file_progress * 0.8)  # Progress from 10% to 90%
        
        update_progress(f"Processing {i+1}/{total_files}: {file_name}", overall_progress)
        
        # Convert to WAV
        wav_file, _ = convert_audio_to_wav(audio_file, temp_dir)
        if wav_file is None:
            results.append(f"❌ Failed to convert {file_name}")
            continue
        
        # Process file
        try:
            diarization, output_lines, unique_speakers, speaker_times = process_diarization(
                pipeline, wav_file, known_speakers
            )
            
            if diarization is None:
                results.append(f"❌ Diarization failed for {file_name}")
                continue
            
            # Transcribe if requested
            transcript_result = None
            combined_output = None
            if transcribe and whisper_model is not None:
                transcript_result = transcribe_audio(whisper_model, wav_file, language, diarization if WHISPERX_AVAILABLE else None)
                if transcript_result and not WHISPERX_AVAILABLE:
                    combined_output = combine_diarization_with_transcript(diarization, transcript_result)
            
            # Clean up temp file
            if wav_file and wav_file != audio_file:
                try:
                    os.remove(wav_file)
                except:
                    pass
            
            # Format output
            header = []
            header.append(f"📊 Detected {len(unique_speakers)} speaker(s) in {file_name}")
            
            sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
            for speaker, duration in sorted_speakers:
                percentage = duration / sum(speaker_times.values()) * 100
                header.append(f"  - {speaker}: {duration:.1f}s ({percentage:.1f}%)")
            
            # Determine which output to use
            if transcript_result:
                if WHISPERX_AVAILABLE:
                    # Format WhisperX results which already include speaker information
                    whisperx_output = []
                    for segment in transcript_result["segments"]:
                        start_time = segment.get("start", 0)
                        end_time = segment.get("end", 0)
                        start_str = str(timedelta(seconds=round(start_time)))
                        end_str = str(timedelta(seconds=round(end_time)))
                        
                        # Check if speaker information is available
                        if "speaker" in segment:
                            speaker = segment["speaker"]
                            text = segment.get("text", "").strip()
                            whisperx_output.append(f"{start_str} - {end_str} {speaker}: {text}")
                        else:
                            text = segment.get("text", "").strip()
                            whisperx_output.append(f"{start_str} - {end_str}: {text}")
                    
                    transcript = "\n".join(header) + "\n\n" + "\n".join(whisperx_output)
                elif combined_output:
                    # Use manually combined output for standard Whisper
                    transcript = "\n".join(header) + "\n\n" + "\n".join(combined_output)
                else:
                    # Fallback to basic diarization
                    transcript = "\n".join(header) + "\n\n" + "\n".join(output_lines)
            else:
                # No transcription, use basic diarization
                transcript = "\n".join(header) + "\n\n" + "\n".join(output_lines)
            
            # Save to file
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            out_file = os.path.join(output_path, f"{base_name}_diarization.txt")
            with open(out_file, "w") as f:
                f.write(transcript)
            
            # Export JSON if requested
            if export_json:
                file_info = {"path": audio_file}
                json_data = format_results_as_json(diarization, transcript_result, file_info)
                json_file = save_json_output(json_data, output_path, base_name)
                
                if json_file:
                    results.append(f"✅ Processed {file_name} - saved to {out_file} and {json_file}")
                else:
                    results.append(f"✅ Processed {file_name} - saved to {out_file}")
            else:
                results.append(f"✅ Processed {file_name} - saved to {out_file}")
            
        except Exception as e:
            results.append(f"❌ Error processing {file_name}: {str(e)}")
    
    update_progress("Batch processing complete", 1.0)
    result_text = f"Processed {total_files} files\n\n" + "\n".join(results)
    return result_text, output_path

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
                            value=(WHISPERX_AVAILABLE or WHISPER_AVAILABLE), 
                            info="Generate text transcription using WhisperX (preferred) or standard Whisper"
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
                    value=(WHISPERX_AVAILABLE or WHISPER_AVAILABLE), 
                    info="Generate text transcription using WhisperX (preferred) or standard Whisper"
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

        # Enable/disable transcription based on Whisper/WhisperX availability
        def update_transcription_availability(value):
            if value and not (WHISPERX_AVAILABLE or WHISPER_AVAILABLE):
                return gr.Checkbox(value=False, interactive=False, 
                                  info="Install WhisperX or Whisper to enable transcription: pip install git+https://github.com/m-bain/whisperx.git")
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
                    - Whisper: {"Available ✅" if WHISPER_AVAILABLE else "Not Available ❌"} (Fallback)
                    
                    #### Installation
                    If transcription is not available, install WhisperX (recommended):
                    ```
                    pip install git+https://github.com/m-bain/whisperx.git
                    ```
                    
                    Or standard Whisper:
                    ```
                    pip install openai-whisper
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
    elif WHISPER_AVAILABLE:
        print("✅ Standard Whisper is installed")
        print("⚠️ For better results, consider installing WhisperX: pip install git+https://github.com/m-bain/whisperx.git")
    else:
        print("⚠️ No transcription system is installed. To enable speech transcription, run:")
        print("   pip install git+https://github.com/m-bain/whisperx.git  # Recommended")
        print("   or: pip install openai-whisper  # Basic")
    
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
