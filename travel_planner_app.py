import os
import re
import json
import time
import requests
import streamlit as st
from urllib.parse import quote
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

load_dotenv()

# ===========================
# CONFIGURATION
# ===========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0.6,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

st.set_page_config(page_title="AI Travel Planner Agent", page_icon="🌍", layout="wide")

# ===========================
# HELPERS
# ===========================
def parse_llm_json(text):
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return [{"day": 1, "activities": [{"time": "All day", "place_name": "Unknown", "description": text}], "notes": ""}]


def get_weather(city):
    try:
        url = f"https://wttr.in/{quote(city)}?format=j1"
        data = requests.get(url, timeout=5).json()
        cond = data["current_condition"][0]
        desc = cond["weatherDesc"][0]["value"]
        temp = cond["temp_C"]
        return f"{desc}, {temp}°C"
    except Exception:
        return "Weather data unavailable"


def review_itinerary(itinerary_json):
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel expert. Review and fix the following itinerary for realism, logical flow, and accurate place naming."),
        ("human", f"Here is the itinerary:\n{json.dumps(itinerary_json, indent=2)}")
    ])
    try:
        response = llm.invoke(review_prompt.format_messages())
        revised = parse_llm_json(response.content)
        return revised if isinstance(revised, list) else itinerary_json
    except Exception:
        return itinerary_json



def create_itinerary(city, days, interests):
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a professional **travel planner and local guide**. 
        Create a **realistic, well-paced {days}-day travel itinerary** for **{city}**, 
        tailored to the traveler's **interests**: {interests}.

        ### Guidelines:
        1. **Realism & Local Knowledge**
        - Use **real, well-known locations** and verified attractions in {city}.
        - Avoid generic names or fictional places.
        - Reflect the **local culture, geography, and travel flow**.
        - Consider **opening hours**, **peak times**, and **local customs** 
            (e.g., lunch around 1 PM, dinner after 7 PM, markets closing by 9 PM).

        2. **Daily Schedule Structure**
        - Start around **8:00–9:00 AM** and end by **9:00–10:00 PM**.
        - Include **3–5 key activities per day**, balanced between sightseeing, food, rest, and exploration.
        - Ensure the **route is geographically logical** (no back-and-forth across the city).
        - Add short **travel breaks or meal stops** between activities.

        3. **Activity Details (Mandatory Fields)**
        Each activity must include:
        - `"time"`: realistic local time (e.g., "10:30 AM")
        - `"place_name"`: specific location (museum, park, restaurant, etc.)
        - `"category"`: one of ["Sightseeing", "Food", "Shopping", "Culture", "Nature", "Nightlife", "Adventure"]
        - `"description"`: 3–4 sentences describing what to see/do and why it fits {interests}

        4. **Notes Per Day**
        Add a `"notes"` field with short, useful local advice such as:
        - Best transport method or ticket info
        - Weather or clothing tips
        - Cultural etiquette or timing suggestions

        5. **Output Format**
        Return output as **strictly valid JSON only**, in this structure:
        [
            {{
                "day": 1,
                "activities": [
                    {{
                        "time": "09:00 AM",
                        "place_name": "Gateway of India",
                        "category": "Sightseeing",
                        "description": "Start your trip at the historic monument overlooking the Arabian Sea..."
                    }},
                    ...
                ],
                "notes": "Use a ferry pass early to avoid queues; great photo spot at sunrise."
            }},
            ...
        ]

        6. **Rules**
        - DO NOT include any text, commentary, or markdown outside the JSON.
        - Avoid repeating places across days unless they serve a new purpose (e.g., a different restaurant in the same area).
        - The final itinerary should feel **authentic, local, and logically ordered**.
        """

    ),
    ("human", "Generate the itinerary.")
    ],template_format="f-string")

    response = llm.invoke(prompt.format_messages(city=city, days=days, interests=interests))
    itinerary_json = parse_llm_json(response.content)

    # ✅ Ensure output is a list of dicts
    if isinstance(itinerary_json, dict):
        itinerary_json = [itinerary_json]
    elif isinstance(itinerary_json, str):
        itinerary_json = [{
            "day": 1,
            "activities": [{"time": "All Day", "place_name": "Unknown", "category": "Sightseeing", "description": itinerary_json}],
            "notes": ""
        }]
    elif not isinstance(itinerary_json, list):
        itinerary_json = [{
            "day": 1,
            "activities": [{"time": "All Day", "place_name": "Unknown", "category": "Sightseeing", "description": "Invalid response format."}],
            "notes": ""
        }]

    # Add weather safely
    weather = get_weather(city)
    for day in itinerary_json:
        if isinstance(day, dict):
            day["notes"] = (day.get("notes", "") + f" | Weather: {weather}").strip(" |")

    # Review and clean
    itinerary_json = review_itinerary(itinerary_json)
    return itinerary_json


# ===========================
# ✨ DISPLAY HELPERS (Side-by-Side Layout)
# ===========================
def display_itinerary_day(day, city):
    """Display itinerary for one day"""
    st.markdown(f"### 📅 Day {day.get('day', '?')}")
    for act in day.get("activities", []):
        place = act.get("place_name", "Unknown Place")
        time_slot = act.get("time", "")
        desc = act.get("description", "")
        category = act.get("category", "")

        maps_url = f"https://www.google.com/maps/search/?api=1&query={quote(place + ', ' + city)}"

        st.markdown(
            f"""
            <div style="margin-bottom: 10px; padding: 10px; border-radius: 10px; background-color: #f8f9fa;">
                <b>🕒 {time_slot}</b> — <a href="{maps_url}" target="_blank"><b>{place}</b></a><br>
                <span style="color: #555;">{desc}</span><br>
                <i style="color: #007bff;">Category: {category}</i>
            </div>
            """,
            unsafe_allow_html=True
        )

    if day.get("notes"):
        st.info(f"💡 {day['notes']}")
    st.divider()


def display_map_day(day, city):
    """Display map for one day"""
    geolocator = Nominatim(user_agent="travel_planner_v2")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]

    try:
        city_loc = geocode(city)
        fmap = folium.Map(location=[city_loc.latitude, city_loc.longitude], zoom_start=12)
    except:
        fmap = folium.Map(location=[19.0760, 72.8777], zoom_start=12)

    for act in day.get("activities", []):
        place = act.get("place_name", "")
        desc = act.get("description", "")
        if not place:
            continue

        try:
            loc = geocode(f"{place}, {city}")
            if loc:
                folium.Marker(
                    [loc.latitude, loc.longitude],
                    popup=folium.Popup(f"<b>{place}</b><br>{desc}", max_width=250),
                    tooltip=place,
                    icon=folium.Icon(color=colors[(day["day"] - 1) % len(colors)], icon="map-marker")
                ).add_to(fmap)
        except:
            continue

    st.components.v1.html(fmap._repr_html_(), height=500)


# ===========================
# ✨ STREAMLIT UI
# ===========================
st.title("🌍 AI Travel Planner Agent")
st.markdown("Plan smarter, travel better — your AI-powered itinerary companion.")

with st.container():
    st.markdown("### 🧳 Enter Your Trip Details")
    col1, col2 = st.columns([1, 1])
    with col1:
        city = st.text_input("Destination City", placeholder="e.g., Tokyo, Mumbai, Paris")
        days = st.slider("Trip Duration (days)", 1, 7, 3)
    with col2:
        interests = st.text_input("Your Interests (comma-separated)", placeholder="e.g., food, culture, nature")

if st.button("✨ Generate My Itinerary", use_container_width=True):
    if not city.strip():
        st.error("❌ Please enter a valid city name.")
    else:
        with st.spinner(f"Creating your {days}-day plan for {city}..."):
            itinerary = create_itinerary(city, days, interests)

        st.success(f"🎉 Your {days}-day itinerary for **{city}** is ready!")
        st.markdown("---")

        # Display each day in a split layout
        for day in itinerary:
            st.markdown(
                f"""
                <div style="margin-top: 40px;">
                    <h2 style="font-size: 28px; font-weight: 700;">🥇 Day {day.get('day', '?')}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns([1.3, 1])

            with col1:
                for act in day.get("activities", []):
                    place = act.get("place_name", "Unknown Place")
                    time_slot = act.get("time", "")
                    desc = act.get("description", "")
                    category = act.get("category", "")
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={quote(place + ', ' + city)}"

                    st.markdown(
                        f"""
                        <div style="
                            background-color: #1e1e1e;
                            border-radius: 12px;
                            padding: 16px;
                            margin-bottom: 15px;
                            border: 1px solid #333;
                        ">
                            <div style="color: #bbb; font-size: 13px;">🕒 {time_slot}</div>
                            <div style="font-size: 17px; font-weight: 600; margin-top: 2px;">
                                <a href="{maps_url}" target="_blank" style="color: #4da3ff; text-decoration: none;">
                                    {place}
                                </a>
                            </div>
                            <div style="color: #ccc; margin-top: 5px;">{desc}</div>
                            <div style="color: #00bfff; font-size: 13px; margin-top: 5px;">
                                🏷️ {category}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if day.get("notes"):
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #102A43;
                            border-left: 5px solid #00BFFF;
                            color: #E6F1FF;
                            padding: 12px;
                            border-radius: 10px;
                            margin-top: 10px;
                        ">
                            💡 {day['notes']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col2:
                display_map_day(day, city)

            st.markdown("---")
