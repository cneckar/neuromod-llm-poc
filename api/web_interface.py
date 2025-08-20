"""
Neuromodulation Web Interface
Streamlit app for interacting with the neuromodulation API
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any

# Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

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
        
        # Pack selection
        st.subheader("Pack Selection")
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
                        st.write(f"**Effects:** {len(pack_details['effects'])} effects")
                        
                        # Show effects
                        with st.expander("View Effects"):
                            for effect in pack_details["effects"]:
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
        
        # Generation parameters
        st.subheader("Generation Parameters")
        max_tokens = st.slider("Max Tokens", 10, 500, 100)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.1)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat Interface")
        
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
                        "top_p": top_p
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
                        st.info(f"Generated {result['tokens_generated']} tokens in {result['generation_time']:.2f}s")
                        
                    else:
                        st.error(f"Generation failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        st.subheader("Quick Generation")
        
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
                                "top_p": top_p
                            },
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.write("**Generated Text:**")
                        st.write(result["generated_text"])
                        st.info(f"Generated in {result['generation_time']:.2f}s")
                    else:
                        st.error(f"Generation failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # System status
        st.subheader("System Status")
        try:
            status_response = requests.get(f"{API_BASE_URL}/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()
                st.write(f"**Model Loaded:** {'✅' if status['model_loaded'] else '❌'}")
                st.write(f"**Current Pack:** {status['current_pack'] or 'None'}")
                st.write(f"**Available Packs:** {status['available_packs']}")
                st.write(f"**Available Effects:** {status['available_effects']}")
            else:
                st.error("Failed to get status")
        except Exception as e:
            st.error(f"Status error: {e}")

if __name__ == "__main__":
    main()
