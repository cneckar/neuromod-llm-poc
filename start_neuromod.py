#!/usr/bin/env python3
"""
Neuromod LLM Startup Script
Automatically handles Hugging Face authentication and starts the chat interface
"""

import os
import sys
from dotenv import load_dotenv

def setup_hf_auth():
    """Setup Hugging Face authentication"""
    # Load environment variables
    load_dotenv()
    
    # Check if token is available
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        try:
            from huggingface_hub import login
            login(token)
            print("✅ Hugging Face authentication successful")
            return True
        except Exception as e:
            print(f"⚠️ Hugging Face authentication failed: {e}")
            return False
    else:
        print("⚠️ No Hugging Face token found in .env file")
        return False

def main():
    """Main startup function"""
    print("🚀 Neuromod LLM Startup")
    print("=" * 30)
    
    # Setup authentication
    setup_hf_auth()
    
    # Start the chat interface
    print("\n🎯 Starting chat interface...")
    try:
        from demo.chat import main as chat_main
        chat_main()
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

