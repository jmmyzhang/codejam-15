#!/usr/bin/env python3
"""
WaveSL - Real-time ASL to Speech Translation
Main application that captures video, processes ASL, and outputs video/audio
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional
from pathlib import Path

from asl_recognizer import ASLRecognizer
from tts_engine import TTSEngine
from audio_output import AudioOutput


class WaveSLApp:
    """Main application class for WaveSL"""
    
    def __init__(self, camera_index: int = 0, audio_device: Optional[str] = None, 
                 model_path: Optional[str] = None):
        """
        Initialize WaveSL application
        
        Args:
            camera_index: Index of the physical camera to use
            audio_device: Name of the audio output device (BlackHole). If None, uses default.
            model_path: Path to trained ASL model (.pt file). If None, uses placeholder.
        """
        self.camera_index = camera_index
        self.cap = None
        
        # Initialize components
        # Default to WLASL pre-trained model if no model path specified
        if model_path is None:
            # Try to find WLASL pre-trained model in default location
            default_model = Path('models/wlasl/best_model.pt')
            if default_model.exists():
                model_path = str(default_model)
                print(f"Using pre-trained WLASL model: {model_path}")
            else:
                print("No pre-trained WLASL model found.")
                print("To train a model, see SETUP_WLASL.md or run:")
                print("  ./train_wlasl_model.sh")
                print("Continuing with placeholder recognition...")
        
        self.asl_recognizer = ASLRecognizer(model_path=model_path)
        self.tts_engine = TTSEngine()
        self.audio_output = AudioOutput(device_name=audio_device)
        
        # Threading and queues
        self.video_queue = queue.Queue(maxsize=2)
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=5)
        
        self.running = False
        self.video_thread = None
        self.processing_thread = None
        self.audio_thread = None
        
    def start(self):
        """Start the application"""
        print("Starting WaveSL...")
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        # Start threads
        self.video_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.audio_thread = threading.Thread(target=self._audio_output_loop, daemon=True)
        
        self.video_thread.start()
        self.processing_thread.start()
        self.audio_thread.start()
        
        print("WaveSL started successfully!")
        print("Video output is available for OBS to capture")
        print("Audio output is being sent to the selected audio device")
        print("Press 'q' to quit")
        
        # Main loop - display video for OBS to capture
        self._display_loop()
        
    def _video_capture_loop(self):
        """Capture frames from camera and add to queue"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Add frame to queue (non-blocking)
            try:
                self.video_queue.put_nowait(frame.copy())
            except queue.Full:
                # Drop oldest frame if queue is full
                try:
                    self.video_queue.get_nowait()
                    self.video_queue.put_nowait(frame.copy())
                except queue.Empty:
                    pass
            
            time.sleep(1/30)  # ~30 FPS
    
    def _processing_loop(self):
        """Process frames for ASL recognition and generate speech"""
        last_text = ""
        last_text_time = 0
        text_accumulation_window = 0.5  # Accumulate signs for 0.5 seconds before processing
        
        while self.running:
            try:
                # Get frame from queue
                frame = self.video_queue.get(timeout=0.1)
                
                # Process frame for ASL recognition
                text = self.asl_recognizer.process_frame(frame)
                
                current_time = time.time()
                
                # Accumulate text over a short window
                if text and text != last_text:
                    if current_time - last_text_time > text_accumulation_window:
                        # Process accumulated text
                        if last_text:
                            print(f"Recognized: {last_text}")
                            # Generate speech
                            audio_data = self.tts_engine.synthesize(last_text)
                            if audio_data is not None:
                                try:
                                    self.audio_queue.put_nowait(audio_data)
                                except queue.Full:
                                    # Drop oldest audio if queue is full
                                    try:
                                        self.audio_queue.get_nowait()
                                        self.audio_queue.put_nowait(audio_data)
                                    except queue.Empty:
                                        pass
                        
                        last_text = text
                        last_text_time = current_time
                    else:
                        # Update accumulating text
                        last_text = text
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _audio_output_loop(self):
        """Output audio to the selected device"""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                self.audio_output.play(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio output loop: {e}")
    
    def _display_loop(self):
        """Display video frames (for OBS to capture)"""
        while self.running:
            try:
                # Get latest frame
                frame = self.video_queue.get(timeout=0.1)
                
                # Display frame in a window (OBS can capture this window)
                cv2.imshow('WaveSL - Camera Feed (for OBS)', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in display loop: {e}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop the application"""
        print("Stopping WaveSL...")
        self.running = False
        
        if self.video_thread:
            self.video_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.audio_output.close()
        print("WaveSL stopped.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WaveSL - Real-time ASL to Speech Translation')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--audio-device', type=str, default=None, 
                       help='Audio output device name (e.g., "BlackHole 2ch" for BlackHole). Use --list-audio to see available devices.')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained ASL model (.pt file). If not provided, uses placeholder recognition.')
    parser.add_argument('--list-audio', action='store_true', 
                       help='List available audio output devices and exit')
    
    args = parser.parse_args()
    
    # List audio devices if requested
    if args.list_audio:
        import sounddevice as sd
        print("\nAvailable audio output devices:")
        print(sd.query_devices())
        return
    
    # Create and run application
    app = WaveSLApp(camera_index=args.camera, audio_device=args.audio_device, 
                    model_path=args.model)
    
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.stop()


if __name__ == '__main__':
    main()

