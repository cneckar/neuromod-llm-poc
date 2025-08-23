"""
Neuromodulation Web Interface
Streamlit app for interacting with the neuromodulation API
Supports both local models and Vertex AI endpoints
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any

# Configuration
try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
except:
    API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="Neuromodulation Interface",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Neuromodulation Interface")
    st.markdown("Apply psychoactive substance analogues to language models")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Check API status
        if st.button("Check API Status"):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API is healthy")
                else:
                    st.error("❌ API is not responding")
            except Exception as e:
                st.error(f"❌ Cannot connect to API: {e}")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        
        # Check if Vertex AI is available
        try:
            status_response = requests.get(f"{API_BASE_URL}/model/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()
                vertex_ai_available = status.get("vertex_ai", {}).get("available", False)
                
                if vertex_ai_available:
                    st.success("☁️ Vertex AI is available")
                    
                    # Get Vertex AI models
                    try:
                        models_response = requests.get(f"{API_BASE_URL}/vertex-ai/models", timeout=5)
                        if models_response.status_code == 200:
                            models_data = models_response.json()
                            vertex_models = models_data.get("models", [])
                            
                            if vertex_models:
                                st.write("**Available Vertex AI Models:**")
                                for model in vertex_models[:5]:  # Show first 5
                                    st.write(f"• {model['name']}")
                                    st.write(f"  {model['description']}")
                                
                                if len(vertex_models) > 5:
                                    st.write(f"... and {len(vertex_models) - 5} more")
                        else:
                            st.warning("Could not load Vertex AI models")
                    except Exception as e:
                        st.warning(f"Vertex AI models error: {e}")
                else:
                    st.info("☁️ Vertex AI is not available")
            else:
                st.error("Failed to get model status")
        except Exception as e:
            st.error(f"Status error: {e}")
        
        # Pack selection
        st.subheader("🧪 Pack Selection")
        try:
            packs_response = requests.get(f"{API_BASE_URL}/packs", timeout=5)
            if packs_response.status_code == 200:
                packs = packs_response.json()
                pack_names = [pack["name"] for pack in packs]
                
                selected_pack = st.selectbox(
                    "Select a pack",
                    ["None"] + pack_names,
                    help="Choose a neuromodulation pack to apply"
                )
                
                if selected_pack != "None":
                    # Get pack details
                    pack_details = next((p for p in packs if p["name"] == selected_pack), None)
                    if pack_details:
                        st.write(f"**Description:** {pack_details['description']}")
                        if "effects" in pack_details:
                            st.write(f"**Effects:** {len(pack_details['effects'])} effects")
                            
                            # Show effects
                            with st.expander("View Effects"):
                                for effect in pack_details["effects"]:
                                    if isinstance(effect, str):
                                        st.write(f"- {effect}")
                                    elif isinstance(effect, dict):
                                        st.write(f"- {effect.get('effect', 'Unknown')} (weight: {effect.get('weight', 'N/A')})")
                
                # Apply pack button
                if st.button("Apply Pack") and selected_pack != "None":
                    with st.spinner(f"Applying {selected_pack}..."):
                        response = requests.post(f"{API_BASE_URL}/packs/{selected_pack}/apply")
                        if response.status_code == 200:
                            st.success(f"✅ Applied {selected_pack}")
                        else:
                            st.error(f"❌ Failed to apply pack: {response.text}")
                            
            else:
                st.error("Failed to load packs")
        except Exception as e:
            st.error(f"Error loading packs: {e}")
        
        # Clear pack
        if st.button("Clear Pack"):
            try:
                response = requests.post(f"{API_BASE_URL}/packs/clear")
                if response.status_code == 200:
                    st.success("✅ Pack cleared")
                else:
                    st.error("❌ Failed to clear pack")
            except Exception as e:
                st.error(f"Error clearing pack: {e}")
        
        # Emotion tracking controls
        st.subheader("🎭 Emotion Tracking")
        
        # Get emotion summary
        if st.button("Get Emotion Summary"):
            try:
                response = requests.get(f"{API_BASE_URL}/emotions/summary", timeout=10)
                if response.status_code == 200:
                    summary = response.json()
                    if "error" not in summary:
                        st.write("**Emotion Summary:**")
                        st.write(f"• Total assessments: {summary.get('total_assessments', 0)}")
                        st.write(f"• Overall valence: {summary.get('valence_trend', 'neutral')}")
                        
                        emotion_changes = summary.get('emotion_changes', {})
                        if emotion_changes:
                            st.write("**Changes:**")
                            for emotion, counts in emotion_changes.items():
                                up_count = counts.get('up', 0)
                                down_count = counts.get('down', 0)
                                if up_count > 0 or down_count > 0:
                                    st.write(f"• {emotion.capitalize()}: {up_count} up, {down_count} down")
                    else:
                        st.warning(summary["error"])
                else:
                    st.error("Failed to get emotion summary")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Export emotions
        if st.button("Export Emotions"):
            try:
                response = requests.post(f"{API_BASE_URL}/emotions/export", timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                    st.info(f"Saved to: {result['filename']}")
                else:
                    st.error("Failed to export emotions")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Generation parameters
        st.subheader("⚙️ Generation Parameters")
        max_tokens = st.slider("Max Tokens", 10, 500, 100)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.1)
        
        # Vertex AI options
        st.subheader("☁️ Vertex AI Options")
        use_vertex_ai = st.checkbox("Use Vertex AI", help="Use Vertex AI instead of local model")
        
        vertex_model = None
        if use_vertex_ai:
            try:
                models_response = requests.get(f"{API_BASE_URL}/vertex-ai/models", timeout=5)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    vertex_models = models_data.get("models", [])
                    
                    if vertex_models:
                        model_names = [model["name"] for model in vertex_models]
                        vertex_model = st.selectbox(
                            "Select Vertex AI Model",
                            model_names,
                            help="Choose which Vertex AI model to use"
                        )
                    else:
                        st.warning("No Vertex AI models available")
                else:
                    st.warning("Could not load Vertex AI models")
            except Exception as e:
                st.warning(f"Vertex AI error: {e}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to discuss?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Prepare chat request
                    chat_request = {
                        "messages": st.session_state.messages,
                        "pack_name": selected_pack if selected_pack != "None" else None,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "use_vertex_ai": use_vertex_ai,
                        "vertex_model": vertex_model
                    }
                    
                    # Send request to API
                    with st.spinner("Generating response..."):
                        response = requests.post(
                            f"{API_BASE_URL}/chat",
                            json=chat_request,
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        assistant_response = result["response"]
                        
                        # Display response
                        message_placeholder.markdown(assistant_response)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        
                        # Show generation info
                        model_info = f"Model: {result.get('model_type', 'unknown')}"
                        if result.get('vertex_model'):
                            model_info += f" ({result['vertex_model']})"
                        
                        st.info(f"{model_info} | Generated {result['tokens_generated']} tokens in {result['generation_time']:.2f}s")
                        
                        # Display emotion data if available
                        if "emotions" in result and result["emotions"]:
                            emotions = result["emotions"]
                            st.subheader("🎭 Emotion Analysis")
                            
                            # Display current emotional state
                            if "current_state" in emotions:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write("**Emotional State:**")
                                    for emotion, state in emotions["current_state"].items():
                                        if state in ["up", "down"]:
                                            st.write(f"• {emotion.capitalize()}: {state}")
                                        elif state == "stable":
                                            st.write(f"• {emotion.capitalize()}: stable")
                                
                                with col2:
                                    st.write("**Overall:**")
                                    st.write(f"• Valence: {emotions.get('valence', 'neutral')}")
                                    st.write(f"• Confidence: {emotions.get('confidence', 0.0):.2f}")
                                
                                with col3:
                                    st.write("**Details:**")
                                    if "timestamp" in emotions:
                                        st.write(f"• Time: {emotions['timestamp']}")
                                    if "note" in emotions:
                                        st.write(f"• Note: {emotions['note']}")
                        
                    else:
                        st.error(f"Generation failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        st.subheader("⚡ Quick Generation")
        
        # Simple text generation
        quick_prompt = st.text_area("Quick prompt", height=100)
        
        if st.button("Generate"):
            if quick_prompt:
                try:
                    with st.spinner("Generating..."):
                        response = requests.post(
                            f"{API_BASE_URL}/generate",
                            params={
                                "prompt": quick_prompt,
                                "pack_name": selected_pack if selected_pack != "None" else None,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                                "top_p": top_p,
                                "use_vertex_ai": use_vertex_ai,
                                "vertex_model": vertex_model
                            },
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.write("**Generated Text:**")
                        st.write(result["generated_text"])
                        
                        model_info = f"Model: {result.get('model_type', 'unknown')}"
                        if result.get('vertex_model'):
                            model_info += f" ({result['vertex_model']})"
                        
                        st.info(f"{model_info} | Generated in {result['generation_time']:.2f}s")
                        
                        # Display emotion data if available
                        if "emotions" in result and result["emotions"]:
                            emotions = result["emotions"]
                            st.subheader("🎭 Emotion Analysis")
                            
                            # Display current emotional state
                            if "current_state" in emotions:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write("**Emotional State:**")
                                    for emotion, state in emotions["current_state"].items():
                                        if state in ["up", "down"]:
                                            st.write(f"• {emotion.capitalize()}: {state}")
                                        elif state == "stable":
                                            st.write(f"• {emotion.capitalize()}: stable")
                                
                                with col2:
                                    st.write("**Overall:**")
                                    st.write(f"• Valence: {emotions.get('valence', 'neutral')}")
                                    st.write(f"• Confidence: {emotions.get('confidence', 0.0):.2f}")
                                
                                with col3:
                                    st.write("**Details:**")
                                    if "timestamp" in emotions:
                                        st.write(f"• Time: {emotions['timestamp']}")
                                    if "note" in emotions:
                                        st.write(f"• Note: {emotions['note']}")
                    else:
                        st.error(f"Generation failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # System status
        st.subheader("📊 System Status")
        try:
            status_response = requests.get(f"{API_BASE_URL}/model/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()
                
                # Local model status
                local_model = status.get("local_model", {})
                st.write(f"**Local Model:** {'✅' if local_model.get('loaded') else '❌'}")
                if local_model.get('loaded'):
                    st.write(f"  • {local_model.get('current_model', 'unknown')}")
                    st.write(f"  • Type: {local_model.get('model_type', 'unknown')}")
                
                # Vertex AI status
                vertex_ai = status.get("vertex_ai", {})
                st.write(f"**Vertex AI:** {'✅' if vertex_ai.get('available') else '❌'}")
                if vertex_ai.get('available'):
                    st.write(f"  • Project: {vertex_ai.get('project_id', 'unknown')}")
                    st.write(f"  • Region: {vertex_ai.get('region', 'unknown')}")
                
                st.write(f"**Emotion Tracking:** ✅ Active")
                
            else:
                st.error("Failed to get status")
        except Exception as e:
            st.error(f"Status error: {e}")

if __name__ == "__main__":
    main()
