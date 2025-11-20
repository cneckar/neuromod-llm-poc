#!/usr/bin/env python3
"""
Setup script for Hugging Face credentials
Required for accessing Llama models
"""

import os
import sys

def setup_hf_credentials():
    """Setup Hugging Face credentials for Llama models"""
    
    print("🔑 Hugging Face Credentials Setup")
    print("=" * 50)
    print()
    print("To use Llama models, you need a Hugging Face token.")
    print("Follow these steps:")
    print()
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token (read access is sufficient)")
    print("3. Copy the token")
    print()
    
    # Get token from user
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return False
    
    # Create .env file
    env_content = f"""# Hugging Face Credentials for Llama Models
HUGGINGFACE_HUB_TOKEN={token}

# Optional: Google Cloud credentials (for Vertex AI)
# GOOGLE_CLOUD_PROJECT_ID=your-project-id
# GOOGLE_CLOUD_REGION=us-central1
# GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with your credentials")
        
        # Set environment variable for current session
        os.environ['HUGGINGFACE_HUB_TOKEN'] = token
        print("✅ Set environment variable for current session")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def test_hf_connection():
    """Test Hugging Face connection"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✅ Successfully connected to Hugging Face as: {user['name']}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Hugging Face: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Neuromod LLM - Hugging Face Setup")
    print("=" * 50)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("📁 .env file already exists")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Keeping existing .env file")
            return
    
    # Setup credentials
    if setup_hf_credentials():
        print()
        print("🧪 Testing connection...")
        if test_hf_connection():
            print()
            print("🎉 Setup complete! You can now use Llama models.")
            print()
            print("Next steps:")
            print("1. Run: python demo/chat.py")
            print("2. Select a Llama model (option 4 or 5)")
            print("3. Enjoy neuromodulated conversations!")
        else:
            print()
            print("⚠️ Setup completed but connection test failed.")
            print("You may need to check your token or internet connection.")
    else:
        print()
        print("❌ Setup failed. Please try again.")

if __name__ == "__main__":
    main()

