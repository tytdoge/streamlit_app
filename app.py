import streamlit as st
import os
import re
import json
import requests
from youtube_transcript_api import YouTubeTranscriptApi

# Load API keys and model names from environment variables.
GITHUB_API_KEY = st.secrets.get("GITHUB_API_KEY") 
GITHUB_API_MODEL_NAME = st.secrets.get("GITHUB_API_MODEL_NAME")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
OPENROUTER_API_MODEL_NAME = st.secrets.get("OPENROUTER_API_MODEL_NAME")

import streamlit as st
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI


def answer(system_prompt, user_prompt, model_type="github"):
    """
    Call the appropriate API provider to get a completion.
    """
    if model_type == "github":
        endpoint = "https://models.inference.ai.azure.com"
        model_name=GITHUB_API_MODEL_NAME
        if not GITHUB_API_KEY:
            raise ValueError("Github API key not found")
        token = GITHUB_API_KEY
    elif model_type == "openrouter":
        endpoint = "https://openrouter.ai/api/v1"
        model_name=OPENROUTER_API_MODEL_NAME
        if not OPENROUTER_API_KEY:
            raise ValueError("Openrouter API key not found")
        token = OPENROUTER_API_KEY
    else:
        raise ValueError("Invalid API type")
        
   

    client = OpenAI(base_url=endpoint, api_key=token)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name
    )
    return response.choices[0].message.content

def extract_video_id(url: str) -> str:
    """
    Extracts the YouTube video ID from a URL.
    """
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_youtube_transcript(video_id, language_code):
    # Build the proxy URL with the provided parameters.
    proxy_api_url = f"https://yt.vl.comp.polyu.edu.hk/transcript?password=for_demo&video_id={video_id}&lang={language_code}"
    response = requests.get(proxy_api_url)
    
    if response.status_code != 200:
        return f"Error fetching transcript: HTTP {response.status_code}"
    
    try:
        transcript_data = response.json()
        transcript = " ".join([entry["text"] for entry in transcript_data])
        return transcript
    except Exception as e:
        # If JSON parsing fails, fallback to returning the raw text.
        return response.text


def main():
    st.title("YouTube Video Summary Generator")
    st.write("Enter a YouTube URL to generate a summary of the video's transcript.")
    

    # Two-column layout: left for inputs, right for output.
    col1, col2 = st.columns(2)
    with col1:
        st.header("Input")
        youtube_url = st.text_input("Enter YouTube URL:")
        language = st.selectbox("Select summary language:", 
                                ["English", "Spanish", "French", "German", "Traditional Chinese"])
        api_provider = st.selectbox("Select API Provider:", ["GitHub Model", "Openrouter"])
        generate_button = st.button("Generate Summary")
    video_id = extract_video_id(youtube_url)
    # Map the selected language to a language code required by the proxy API.
    language_map = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Traditional Chinese": "zh-TW"
    }

    language_code = language_map.get(language, "en")


    
        
    with col2:
        st.header("Summary Output")
        summary_placeholder = st.empty()
        with st.expander("View AI Prompt", expanded=False):
            st.write("The following prompt structure is sent to the API:")
            st.code(
                "System prompt: 'You are an AI summarization assistant.'\n"
                "User prompt: 'Please summarize the following YouTube transcript in <language>:\\n\\n<transcript>'",
                language="python"
            )
            
    if generate_button:
        if not youtube_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Could not extract a video ID from the URL. Please check the URL format.")
            else:
                with st.spinner("Fetching transcript..."):
                    transcript = get_youtube_transcript(video_id,language_code)
                if transcript.startswith("Error"):
                    st.error(transcript)
                else:
                    st.success("Transcript fetched successfully!")
                    system_prompt = "You are an AI summarization assistant."
                    user_prompt = f"Please summarize the following YouTube transcript in {language}:\n\n{transcript}"
                    try:
                        with st.spinner("Generating summary..."):
                            model_type = "github" if api_provider == "GitHub Model" else "openrouter"
                            summary = answer(system_prompt, user_prompt, model_type=model_type)
                        summary_placeholder.text_area("Generated Summary", summary, height=300)
                    except Exception as e:
                        st.error(f"Error during API call: {e}")

if __name__ == "__main__":
    main()
