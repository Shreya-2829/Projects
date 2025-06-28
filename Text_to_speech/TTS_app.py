import streamlit as st
import pyttsx3
import os
import base64

# pyttsx3: A text-to-speech engine that works offline.
# os: Provides access to operating system functions, like deleting files.
# base64: A module that helps us encode binary data (like audio) for download links in web format.


# Title
st.markdown("<h1 style='text-align: center;'>🗣️ Type2Talk</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>From keyboard to voice — instantly.</p>", unsafe_allow_html=True)

# Input text
text = st.text_area("Enter text to convert to speech", height=200)

# Voice selection
voice_type = st.radio("Choose Voice Type", ["Male", "Female"])

# Speed
rate = st.slider("Select Speech Speed (words per minute)", 100, 200, 150)

# Volume
volume = st.slider("Select Volume", 0.0, 1.0, 0.8)
# 0.0 to 1.0: Range of volume in pyttsx3 (0 = mute, 1 = full volume).
# 0.8: Default volume is 80%.

if st.button("Convert to Audio"):
    if text.strip() == "": # text.strip(): Removes spaces or newlines from start/end.
        st.warning("Please enter some text.")
    else:
        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        # pyttsx3.init(): Starts the speech engine.
        # engine: Stores the TTS engine so we can use it to set voice, rate, volume, etc.

        engine.setProperty('rate', rate)     # Speed of speech.
        engine.setProperty('volume', volume) # Loudness of the voice.


        # Get available voices
        voices = engine.getProperty('voices')
        if voice_type == "Male":
            engine.setProperty('voice', voices[0].id)
        else:
            engine.setProperty('voice', voices[1].id)

        '''
            getProperty('voices'): Returns a list of available system voices.
            voices[0]: Usually the male voice.
            voices[1]: Usually the female voice.
            .id: The unique ID of the voice to pass to setProperty('voice', ...).
        '''

        # Save to file
        output_file = "converted_speech.mp3" # Sets filename for saving.
        engine.save_to_file(text, output_file)  # Converts text to speech and saves it to an audio file
        engine.runAndWait() # Executes the queued speech commands. Must be called.
 
        # Read file and create download link
        with open(output_file, "rb") as f:  # Opens the audio file in binary mode ("rb" = read binary).
            audio_bytes = f.read()

        '''
        WITH: A context manager in Python.
            A Python keyword used to open resources safely, like files.
            It automatically closes the file after you're done with it (even if an error occurs).
            Saves you from writing f.close() manually.
        '''

        st.audio(audio_bytes, format='audio/mp3') # Adds an audio player to the webpage.

        b64 = base64.b64encode(audio_bytes).decode() 
        # base64.b64encode(...): A method that encodes binary data into base64 format.
        # .decode(): Converts the base64 bytes into a normal UTF-8 string (text), so you can use it in HTML.
        # Converts the audio into base64 so it can be embedded in an HTML download link.

        href = f'<a href="data:audio/mp3;base64,{b64}" download="converted_speech.mp3">📥 Download Audio</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Clean up file
        os.remove(output_file)  # Deletes the audio file after it’s used.
