import os
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()


def test_azure_auth():
    """Test Azure OpenAI authentication"""
    try:
        # Test with a simple completion
        response = completion(
            model="azure/gpt-4.1",
            messages=[{"role": "user", "content": "Hello, this is a test message."}],
            max_tokens=10,
        )
        print("✅ Authentication successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Azure OpenAI authentication...")
    print(f"AZURE_API_KEY: {'Set' if os.getenv('AZURE_API_KEY') else 'Not set'}")
    print(f"AZURE_API_BASE: {os.getenv('AZURE_API_BASE', 'Not set')}")
    print(f"AZURE_API_VERSION: {os.getenv('AZURE_API_VERSION', 'Not set')}")
    print("-" * 50)

    test_azure_auth()
