"""
Test script to verify Part 1: Project Setup & Configuration
Run this to ensure everything is set up correctly before proceeding to Part 2
"""

import sys
from pathlib import Path

def test_directory_structure():
    """Test if all required directories exist"""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        "data/uploads",
        "data/fetched/pdfs",
        "data/fetched/metadata",
        "data/processed",
        "data/faiss_index",
        "src/search",
        "src/download",
        "prompts"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_config_file():
    """Test if config.py can be imported and validated"""
    print("\n🔍 Testing configuration file...")
    
    try:
        import config
        print("  ✅ config.py imported successfully")
        
        # Test validation
        issues = config.validate_config()
        if issues:
            print("  ⚠️  Configuration warnings:")
            for issue in issues:
                print(f"     {issue}")
        else:
            print("  ✅ Configuration is valid")
        
        # Print summary
        print("\n  📋 Configuration Summary:")
        summary = config.get_config_summary()
        for key, value in summary.items():
            print(f"     {key}: {value}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed to import config: {e}")
        return False

def test_env_file():
    """Test if .env file exists"""
    print("\n🔍 Testing environment file...")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_example_path.exists():
        print("  ✅ .env.example exists")
    else:
        print("  ❌ .env.example is missing")
    
    if env_path.exists():
        print("  ✅ .env file exists")
        return True
    else:
        print("  ⚠️  .env file not found")
        print("     Create one by copying .env.example:")
        print("     cp .env.example .env")
        print("     Then add your API keys to the .env file")
        return False

def test_requirements():
    """Test if requirements.txt exists"""
    print("\n🔍 Testing requirements file...")
    
    req_path = Path("requirements.txt")
    if req_path.exists():
        print("  ✅ requirements.txt exists")
        with open(req_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"  📦 {len(lines)} packages listed")
        return True
    else:
        print("  ❌ requirements.txt is missing")
        return False

def test_python_version():
    """Test Python version"""
    print("\n🔍 Testing Python version...")
    
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("  ✅ Python version is compatible (3.9+)")
        return True
    else:
        print("  ⚠️  Python 3.9+ recommended, you have {version.major}.{version.minor}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Research RAG Assistant - Setup Verification")
    print("=" * 60)
    
    results = {
        "Directory Structure": test_directory_structure(),
        "Configuration File": test_config_file(),
        "Environment File": test_env_file(),
        "Requirements File": test_requirements(),
        "Python Version": test_python_version()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Setup is complete.")
        print("\n📝 Next steps:")
        print("   1. Create .env file if you haven't: cp .env.example .env")
        print("   2. Add your GROQ_API_KEY to .env")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Ready for Part 2!")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("   Review the error messages and ensure all files are created.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)