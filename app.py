import streamlit as st
import os
import re
import json
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

# Load API keys and model names from Streamlit secrets.
GITHUB_API_KEY = st.secrets.get("GITHUB_API_KEY")
GITHUB_API_MODEL_NAME = st.secrets.get("GITHUB_API_MODEL_NAME")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
OPENROUTER_API_MODEL_NAME = st.secrets.get("OPENROUTER_API_MODEL_NAME")


def answer(system_prompt, user_prompt, model_type="github"):
    """
    Call the appropriate API provider to get a completion.
    """
    if model_type == "github":
        endpoint = "https://models.inference.ai.azure.com"
        model_name = GITHUB_API_MODEL_NAME
        if not GITHUB_API_KEY:
            raise ValueError("Github API key not found")
        token = GITHUB_API_KEY
    elif model_type == "openrouter":
        endpoint = "https://openrouter.ai/api/v1"
        model_name = OPENROUTER_API_MODEL_NAME
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
    Expects URL formats like:
    'https://www.youtube.com/watch?v=VIDEO_ID' or 'https://youtu.be/VIDEO_ID'
    """
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_youtube_transcript(video_id, language_code):
    """
    Fetch the transcript using a proxy service.
    Builds the URL using the video ID and the language code.
    """
    proxy_api_url = (
        f"https://yt.vl.comp.polyu.edu.hk/transcript?password=for_demo"
        f"&video_id={video_id}&lang={language_code}"
    )
    response = requests.get(proxy_api_url)
    
    if response.status_code != 200:
        return f"Error fetching transcript: HTTP {response.status_code}"
    
    try:
        transcript_data = response.json()
        # If JSON parsing works, return the data.
        return transcript_data
    except Exception as e:
        # Fallback: if JSON parsing fails, return the raw text.
        return response.text


def seconds_to_hhmmss(seconds):
    """
    Convert seconds to hh:mm:ss format.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segment_transcript(transcript_data, segment_duration=120):
    """
    Segment transcript entries into sections based on a given segment duration (in seconds).
    Assumes each entry in transcript_data is a dict with a "start" key.
    Returns a list of sections where each section is a dict with:
      - "start": the start time of the section (in seconds)
      - "entries": list of transcript entries for the section
    """
    sections = []
    current_section = []
    current_start = None
    for entry in transcript_data:
        # Get the start time from the entry; default to 0 if not present.
        start = entry.get("start", 0)
        if current_start is None:
            current_start = start
        # If the difference exceeds segment_duration, finalize the current section.
        if start - current_start >= segment_duration:
            if current_section:
                sections.append({"start": current_start, "entries": current_section})
            current_section = [entry]
            current_start = start
        else:
            current_section.append(entry)
    if current_section:
        sections.append({"start": current_start, "entries": current_section})
    return sections


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
        generate_detail_button = st.button("Generate Detailed Summary")
    
    # Map the selected language to a language code required by the proxy API.
    language_map = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Traditional Chinese": "zh-TW"
    }
    language_code = language_map.get(language, "en")
    
    # Simple summary generation (existing functionality)
    if generate_button:
        if not youtube_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Could not extract a video ID from the URL. Please check the URL format.")
            else:
                with st.spinner("Fetching transcript..."):
                    transcript_data = get_youtube_transcript(video_id, language_code)
                    # If transcript_data is a string (error or raw text), wrap it in a list.
                    if isinstance(transcript_data, str):
                        transcript = transcript_data
                    else:
                        transcript = " ".join([entry["text"] for entry in transcript_data])
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
                        st.text_area("Generated Summary", summary, height=300)
                    except Exception as e:
                        st.error(f"Error during API call: {e}")
    
    # Detailed summary generation with sections.
    if generate_detail_button:
        if not youtube_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Could not extract a video ID from the URL. Please check the URL format.")
            else:
                with st.spinner("Fetching transcript..."):
                    transcript_data = get_youtube_transcript(video_id, language_code)
                    if isinstance(transcript_data, str):
                        st.error("Expected transcript data in JSON format, but got raw text.")
                        return
                try:
                    sections = segment_transcript(transcript_data, segment_duration=120)
                except Exception as e:
                    st.error(f"Error segmenting transcript: {e}")
                    return
                st.success("Transcript fetched and segmented successfully!")
                
                # Initialize session state to hold detailed summaries.
                if "detailed_summaries" not in st.session_state:
                    st.session_state["detailed_summaries"] = []
                    system_prompt = "You are an AI summarization assistant."
                    for section in sections:
                        section_start = section["start"]
                        hhmmss = seconds_to_hhmmss(section_start)
                        section_transcript = " ".join([entry["text"] for entry in section["entries"]])
                        user_prompt = f"Please summarize the following section of a YouTube transcript starting at {hhmmss} in {language}:\n\n{section_transcript}"
                        try:
                            model_type = "github" if api_provider == "GitHub Model" else "openrouter"
                            section_summary = answer(system_prompt, user_prompt, model_type=model_type)
                        except Exception as e:
                            section_summary = f"Error generating summary: {e}"
                        st.session_state["detailed_summaries"].append({
                            "start": section_start,
                            "hhmmss": hhmmss,
                            "transcript": section_transcript,
                            "summary": section_summary
                        })
                
                with col2:
                    st.header("Detailed Summary Output")
                    for idx, section in enumerate(st.session_state["detailed_summaries"]):
                        with st.container():
                            # Create a hyperlink header with timestamp.
                            section_url = f"https://www.youtube.com/watch?v={video_id}&t={int(section['start'])}s"
                            st.markdown(f"### [Section starting at {section['hhmmss']}]({section_url})")
                            
                            # Editable summary text area.
                            new_summary = st.text_area(f"Section {idx+1} Summary", section["summary"], key=f"summary_{idx}")
                            if st.button("Save", key=f"save_{idx}"):
                                st.session_state["detailed_summaries"][idx]["summary"] = new_summary
                                st.success("Section summary updated!")
                            
                            # Expander to show the transcript with timestamps.
                            with st.expander("Show transcript"):
                                for entry in sections[idx]["entries"]:
                                    ts = seconds_to_hhmmss(entry.get("start", 0))
                                    st.write(f"[{ts}] {entry.get('text', '')}")
                            
                            # Buttons for additional summary adjustments.
                            btn_cols = st.columns(3)
                            with btn_cols[0]:
                                if st.button("More details", key=f"more_details_{idx}"):
                                    prompt = f"Please provide a more detailed summary for the following transcript section starting at {section['hhmmss']}:\n\n{section['transcript']}"
                                    try:
                                        new_detail = answer(system_prompt, prompt, model_type=("github" if api_provider=="GitHub Model" else "openrouter"))
                                        st.session_state["detailed_summaries"][idx]["summary"] = new_detail
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            with btn_cols[1]:
                                if st.button("More concise", key=f"more_concise_{idx}"):
                                    prompt = f"Please provide a more concise summary for the following transcript section starting at {section['hhmmss']}:\n\n{section['transcript']}"
                                    try:
                                        new_concise = answer(system_prompt, prompt, model_type=("github" if api_provider=="GitHub Model" else "openrouter"))
                                        st.session_state["detailed_summaries"][idx]["summary"] = new_concise
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            with btn_cols[2]:
                                if st.button("More fun", key=f"more_fun_{idx}"):
                                    prompt = f"Please make the following summary more fun with emojis for the transcript section starting at {section['hhmmss']}:\n\n{section['transcript']}"
                                    try:
                                        new_fun = answer(system_prompt, prompt, model_type=("github" if api_provider=="GitHub Model" else "openrouter"))
                                        st.session_state["detailed_summaries"][idx]["summary"] = new_fun
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                    
                    # Download button: generate an HTML version of the detailed summary.
                    html_content = "<html><body>"
                    for section in st.session_state["detailed_summaries"]:
                        section_url = f"https://www.youtube.com/watch?v={video_id}&t={int(section['start'])}s"
                        html_content += f"<h3><a href='{section_url}'>Section starting at {section['hhmmss']}</a></h3>"
                        html_content += f"<p>{section['summary']}</p>"
                    html_content += "</body></html>"
                    st.download_button("Download Summary as HTML", data=html_content, file_name="detailed_summary.html", mime="text/html")


if __name__ == "__main__":
    main()
