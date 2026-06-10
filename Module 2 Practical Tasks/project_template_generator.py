import sys
from pathlib import Path

def generate_ai_template(project_name, modules=[]):
    root = Path(project_name)
    
    base_requirements = [
        "openai>=1.30",
        "python-dotenv>=1.0",
        "pydantic>=2.7"
    ]

    if "rag" in modules:
        print("-> Adding Vector Database and Search extensions...")
        base_requirements.extend(["chromadb>=0.5", "langchain-community>=0.2"])
        
    if "agents" in modules:
        print("-> Adding Multi-Agent workflow modules...")
        base_requirements.extend(["langgraph>=0.1", "crewai>=0.30"])
        
    if "finetuning" in modules:
        print("-> Adding local training libraries...")
        base_requirements.extend(["torch>=2.2", "transformers>=4.40", "peft>=0.10"])

    directories = ["app", "data", "tests"]
    for d in directories:
        (root / d).mkdir(parents=True, exist_ok=True)
        if d == "app":
            (root / d / "__init__.py").touch()
    req_txt = "\n".join(base_requirements)
    (root / "requirements.txt").write_text(req_txt)
    
    (root / ".gitignore").write_text("venv/\n.env\n__pycache__/\n")

    main_code = 'print("AI Project Stack Activated successfully!")\n'
    (root / "app" / "main.py").write_text(main_code)

    print(f"\nSuccess! Standardized template layout '{project_name}' generated.")

if __name__ == "__main__":
    args = sys.argv[1:]
    
    if not args:
        name = "my-custom-ai-app"
        flags = []
    else:
        name = args[0]
        flags = []
        for a in args:
            if "rag" in a:
                flags.append("rag")
            if "agent" in a:
                flags.append("agents")
            if "finetune" in a or "fine-tuning" in a:
                flags.append("finetuning")

    generate_ai_template(name, modules=flags)