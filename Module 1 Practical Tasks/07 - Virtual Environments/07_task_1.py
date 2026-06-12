import os
from pathlib import Path


def create_project(project_name, base_dir="."):
    """Create a new AI project with standard structure."""
    root = Path(base_dir) / project_name

    directories = [
        "",
        "app",
        "app/agents",
        "app/tools",
        "app/models",
        "tests",
        "data",
        "notebooks",
        "scripts",
    ]

    for dir_path in directories:
        (root / dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py in Python packages
        if dir_path.startswith("app"):
            init_file = root / dir_path / "__init__.py"
            init_file.touch()

    # Create requirements.txt
    requirements = """# Core AI
openai>=1.30
anthropic>=0.25
langchain>=0.2
langchain-openai>=0.1

# Backend
fastapi>=0.111
uvicorn>=0.29
pydantic>=2.7

# Utilities
python-dotenv>=1.0
httpx>=0.27
"""
    (root / "requirements.txt").write_text(requirements)

    # Create .gitignore
    gitignore = """venv/
.env
__pycache__/
*.pyc
data/raw/
models/
.vscode/
.DS_Store
"""
    (root / ".gitignore").write_text(gitignore)

    # Create .env template
    env_template = """# API Keys
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Config
MODEL_NAME=gpt-4
TEMPERATURE=0.7
"""
    (root / ".env.example").write_text(env_template)

    # Create main.py
    main_py = '''"""
Main application entry point.
"""
from dotenv import load_dotenv

load_dotenv()

def main():
    print("AI Project ready!")

if __name__ == "__main__":
    main()
'''
    (root / "app" / "main.py").write_text(main_py)

    print(f"Project '{project_name}' created at {root}")
    print(f"\nNext steps:")
    print(f"  cd {project_name}")
    print(f"  python -m venv venv")
    print(f"  source venv/Scripts/activate  # Windows")
    print(f"  pip install -r requirements.txt")
    print(f"  cp .env.example .env  # then add your API keys")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "my-ai-project"
    create_project(name)