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
    Expected URL formats:
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
        return transcript_data
    except Exception as e:
        # If JSON parsing fails, fallback to plain text.
        return response.text


def seconds_to_hhmmss(seconds):
    """
    Convert seconds to hh:mm:ss format.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def hhmmss_to_seconds(hhmmss: str) -> int:
    """
    Convert hh:mm:ss string to seconds.
    """
    parts = hhmmss.split(":")
    if len(parts) != 3:
        return 0
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def segment_summary(summary_text):
    """
    Parses the AI-generated detailed summary into sections.
    Each section is expected to start with a line like:
      "hh:mm:ss - Section Title: summary text..."
    Returns a list of sections (dicts with keys "timestamp" and "summary").
    """
    pattern = re.compile(r"^(\d{2}:\d{2}:\d{2})\s*-\s*(.*)$")
    sections = []
    current_section = None
    for line in summary_text.splitlines():
        match = pattern.match(line)
        if match:
            if current_section is not None:
                sections.append(current_section)
            current_section = {
                "timestamp": match.group(1),
                "summary": match.group(2).strip()
            }
        else:
            if current_section is not None:
                current_section["summary"] += "\n" + line.strip()
    if current_section is not None:
        sections.append(current_section)
    return sections


def filter_transcript_entries(transcript_data, start_sec, end_sec=None):
    """
    Filters the transcript entries (list of dicts) to include those with
    a "start" value >= start_sec and, if provided, < end_sec.
    Returns a string that concatenates these entries with timestamps.
    """
    filtered = []
    for entry in transcript_data:
        entry_start = entry.get("start", 0)
        if entry_start >= start_sec and (end_sec is None or entry_start < end_sec):
            ts = seconds_to_hhmmss(entry_start)
            filtered.append(f"[{ts}] {entry.get('text', '')}")
    return "\n".join(filtered)


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
                    if isinstance(transcript_data, list):
                        transcript = " ".join([entry["text"] for entry in transcript_data])
                    else:
                        transcript = transcript_data
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
    
    # Detailed summary generation (segmenting the AI summary into sections)
    if generate_detail_button:
        if not youtube_url:
            st.error("Please enter a valid YouTube URL.")
            return
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Could not extract a video ID from the URL. Please check the URL format.")
            return
        with st.spinner("Fetching transcript..."):
            transcript_data = get_youtube_transcript(video_id, language_code)
        # If transcript_data is not a list, wrap it in a dummy list.
        if not isinstance(transcript_data, list):
            transcript_data = [{"start": 0, "text": transcript_data}]
        full_transcript = " ".join([entry["text"] for entry in transcript_data])
        
        # Generate a detailed summary from the AI (the summary is divided into sections).
        system_prompt = "You are an AI summarization assistant."
        user_prompt = (
            f"Please generate a detailed summary of the following YouTube transcript in {language}.\n"
            f"Divide the summary into sections. Each section should begin with a timestamp in hh:mm:ss format "
            f"indicating the start time in the video, followed by a dash and a section title, then the summary text.\n\n"
            f"Transcript:\n{full_transcript}"
        )
        try:
            with st.spinner("Generating detailed summary..."):
                model_type = "github" if api_provider == "GitHub Model" else "openrouter"
                detailed_summary_text = answer(system_prompt, user_prompt, model_type=model_type)
        except Exception as e:
            st.error(f"Error during API call: {e}")
            return

        # Parse the detailed summary into sections.
        sections = segment_summary(detailed_summary_text)
        if not sections:
            st.error("Failed to parse detailed summary into sections.")
            return
        st.success("Detailed summary generated successfully!")
        
        # Store parsed sections in session state.
        st.session_state["detailed_summaries"] = sections
        
        with col2:
            st.header("Detailed Summary Output")
            for idx, sec in enumerate(st.session_state["detailed_summaries"]):
                with st.container():
                    # Use the parsed timestamp from the section.
                    timestamp = sec["timestamp"]
                    start_sec = hhmmss_to_seconds(timestamp)
                    section_url = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"
                    st.markdown(f"### [Section starting at {timestamp}]({section_url})")
                    
                    # Editable summary text area.
                    new_summary = st.text_area(f"Section {idx+1} Summary", sec["summary"], key=f"summary_{idx}")
                    if st.button("Save", key=f"save_{idx}"):
                        st.session_state["detailed_summaries"][idx]["summary"] = new_summary
                        st.success("Section summary updated!")
                    
                    # Determine transcript excerpt for this section.
                    next_start = None
                    if idx + 1 < len(st.session_state["detailed_summaries"]):
                        next_timestamp = st.session_state["detailed_summaries"][idx+1]["timestamp"]
                        next_start = hhmmss_to_seconds(next_timestamp)
                    transcript_section = filter_transcript_entries(transcript_data, start_sec, next_start)
                    
                    with st.expander("Show transcript"):
                        st.text_area("Transcript", transcript_section, height=150)
                    
                    # Buttons for additional summary adjustments.
                    btn_cols = st.columns(3)
                    with btn_cols[0]:
                        if st.button("More details", key=f"more_details_{idx}"):
                            prompt = f"Please provide a more detailed summary for the following transcript section starting at {timestamp}:\n\n{transcript_section}"
                            try:
                                new_detail = answer(system_prompt, prompt, model_type=("github" if api_provider=="GitHub Model" else "openrouter"))
                                st.session_state["detailed_summaries"][idx]["summary"] = new_detail
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    with btn_cols[1]:
                        if st.button("More concise", key=f"more_concise_{idx}"):
                            prompt = f"Please provide a more concise summary for the following transcript section starting at {timestamp}:\n\n{transcript_section}"
                            try:
                                new_concise = answer(system_prompt, prompt, model_type=("github" if api_provider=="GitHub Model" else "openrouter"))
                                st.session_state["detailed_summaries"][idx]["summary"] = new_concise
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    with btn_cols[2]:
                        if st.button("More fun", key=f"more_fun_{idx}"):
                            prompt = f"Please make the following summary more fun with emojis for the transcript section starting at {timestamp}:\n\n{transcript_section}"
                            try:
                                new_fun = answer(system_prompt, prompt, model_type=("github" if api_provider=="GitHub Model" else "openrouter"))
                                st.session_state["detailed_summaries"][idx]["summary"] = new_fun
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            
            # Download button: generate an HTML version of the detailed summary.
            html_content = "<html><body>"
            for sec in st.session_state["detailed_summaries"]:
                sec_start = hhmmss_to_seconds(sec["timestamp"])
                sec_url = f"https://www.youtube.com/watch?v={video_id}&t={sec_start}s"
                html_content += f"<h3><a href='{sec_url}'>Section starting at {sec['timestamp']}</a></h3>"
                html_content += f"<p>{sec['summary']}</p>"
            html_content += "</body></html>"
            st.download_button("Download Summary as HTML", data=html_content, file_name="detailed_summary.html", mime="text/html")


if __name__ == "__main__":
    main()
