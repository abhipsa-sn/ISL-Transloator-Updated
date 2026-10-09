"""
Text-to-Speech engine module for Indian Sign Language translation.
Bypasses pyttsx3 and uses Windows native SAPI (SpVoice) directly via COM.
This completely resolves threading and freezing issues on Windows.
"""

import logging
import win32com.client
import pythoncom

try:
    from . import config
except ImportError:
    try:
        import config
    except ImportError:
        config = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSEngine:
    """
    TTS wrapper using Windows SAPI (Speech API) via win32com.
    """

    def __init__(self):
        """Initializes the SAPI voice."""
        try:
            # Initialize COM on the current thread
            pythoncom.CoInitialize()
            self.voice = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Set rate and volume
            rate = getattr(config, 'TTS_RATE', 150) if config else 150
            volume = getattr(config, 'TTS_VOLUME', 1.0) if config else 1.0
            
            # SAPI rate is from -10 to 10 (default 0). Map 150wpm to 0.
            # 100 wpm -> -5, 200 wpm -> 5
            sapi_rate = int((rate - 150) / 10)
            self.set_rate(sapi_rate)
            self.set_volume(int(volume * 100)) # SAPI volume is 0 to 100
            
            logger.info("TTSEngine (SAPI) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SAPI voice: {e}")
            self.voice = None

    def set_rate(self, sapi_rate):
        """Sets the SAPI speaking rate (-10 to 10)."""
        if self.voice:
            try:
                self.voice.Rate = sapi_rate
            except Exception as e:
                logger.error(f"Error setting rate: {e}")

    def set_volume(self, volume):
        """Sets SAPI volume (0 to 100)."""
        if self.voice:
            try:
                self.voice.Volume = volume
            except Exception as e:
                logger.error(f"Error setting volume: {e}")

    def speak(self, text):
        """Speaks the given text asynchronously (non-blocking)."""
        if not self.voice:
            logger.warning("TTS voice not initialized. Cannot speak.")
            return
            
        try:
            # 1 = SVSFlagsAsync (asynchronous speak)
            # 2 = SVSFPurgeBeforeSpeak (interrupt current speech and speak new word immediately)
            # We want to purge current speech if a new word is detected, or queue it.
            # Using 1 (Async) will queue/buffer the words cleanly.
            pythoncom.CoInitialize()
            self.voice.Speak(text, 1)
        except Exception as e:
            logger.error(f"Error during SAPI Speak: {e}")

    def speak_sync(self, text):
        """Speaks the given text in a blocking manner."""
        if not self.voice:
            return
        try:
            pythoncom.CoInitialize()
            self.voice.Speak(text, 0) # 0 = Synchronous
        except Exception as e:
            logger.error(f"Error during SAPI Speak sync: {e}")

    def stop(self):
        """Interrupts any current speech."""
        if self.voice:
            try:
                # Speak empty string with purge flag to stop current speech
                self.voice.Speak("", 2)
            except Exception:
                pass


if __name__ == '__main__':
    import time
    print("Testing TTSEngine (SAPI)...")
    tts = TTSEngine()
    
    print("Speaking 'Hello' asynchronously...")
    tts.speak("Hello")
    
    print("Speaking 'Thank You' asynchronously...")
    tts.speak("Thank You")
    
    time.sleep(3)
    print("Test complete.")
