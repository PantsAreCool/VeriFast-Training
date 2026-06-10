"""
git_workflow.py - Automates common Git operations for AI projects.
Uses subprocess for Git commands and optionally gh CLI for GitHub.

Usage:
    python git_workflow.py init my-ai-project
    python git_workflow.py feature add-rag-pipeline
    python git_workflow.py commit feat "implement RAG pipeline with vector search"
    python git_workflow.py push
    python git_workflow.py pr "Add RAG pipeline" "Implement retrieval-augmented generation"
    python git_workflow.py status
    python git_workflow.py log 10
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    cmd = ["git"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Git error: {result.stderr.strip()}")
        sys.exit(1)
    return result


def run_gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a gh (GitHub CLI) command."""
    cmd = ["gh"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"GitHub CLI error: {result.stderr.strip()}")
        sys.exit(1)
    return result


def init_project(project_name: str):
    """Initialize a new Git repository with AI project structure."""
    project_dir = Path(project_name)
    project_dir.mkdir(exist_ok=True)

    # Create project structure
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "configs").mkdir(exist_ok=True)
    (project_dir / "notebooks").mkdir(exist_ok=True)

    # Create .gitkeep for empty directories
    for d in ["data", "models", "configs", "notebooks"]:
        (project_dir / d / ".gitkeep").touch()

    # Create .gitignore
    gitignore_content = """\
__pycache__/
*.pyc
.venv/
venv/
.env
*.bin
*.safetensors
*.gguf
*.pt
*.pth
data/raw/
data/processed/
mlruns/
wandb/
.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store
"""
    (project_dir / ".gitignore").write_text(gitignore_content)

    # Create README
    readme_content = f"""# {project_name}

AI project initialized on {datetime.now().strftime('%Y-%m-%d')}.

## Structure
- `src/` - Source code
- `tests/` - Test files
- `configs/` - Configuration files
- `notebooks/` - Jupyter notebooks
- `data/` - Data directory (gitignored)
- `models/` - Model weights (gitignored)
"""
    (project_dir / "README.md").write_text(readme_content)

    # Create requirements.txt
    (project_dir / "requirements.txt").write_text(
        "fastapi==0.110.0\nuvicorn==0.27.1\npydantic==2.6.1\n"
    )

    # Initialize Git
    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=project_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "chore: initialize project structure"],
        cwd=project_dir,
        capture_output=True,
    )

    print(f"Project '{project_name}' initialized with Git.")
    print(f"  Directory: {project_dir.resolve()}")
    print(f"  Branch: main")
    print(f"  Initial commit: chore: initialize project structure")


def create_feature_branch(branch_name: str):
    """Create and switch to a feature branch."""
    full_name = f"feature/{branch_name}" if not branch_name.startswith("feature/") else branch_name

    # Make sure we are on main and up to date
    run_git("checkout", "main")
    run_git("pull", "origin", "main", check=False)

    # Create and switch to feature branch
    run_git("checkout", "-b", full_name)
    print(f"Created and switched to branch: {full_name}")


def conventional_commit(commit_type: str, message: str):
    """Create a commit with conventional commit format."""
    valid_types = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "chore", "ci"]
    if commit_type not in valid_types:
        print(f"Invalid commit type: {commit_type}")
        print(f"Valid types: {', '.join(valid_types)}")
        sys.exit(1)

    commit_msg = f"{commit_type}: {message}"

    # Stage all changes
    run_git("add", ".")

    # Check if there are changes to commit
    status = run_git("status", "--porcelain")
    if not status.stdout.strip():
        print("No changes to commit.")
        return

    # Commit
    result = run_git("commit", "-m", commit_msg)
    print(f"Committed: {commit_msg}")
    print(result.stdout.strip())


def push_branch():
    """Push current branch to remote and set upstream."""
    # Get current branch name
    result = run_git("rev-parse", "--abbrev-ref", "HEAD")
    branch = result.stdout.strip()
    run_git("push", "-u", "origin", branch)
    print(f"Pushed branch '{branch}' to origin.")


def create_pr(title: str, body: str):
    """Create a pull request using gh CLI."""
    # Get current branch
    result = run_git("rev-parse", "--abbrev-ref", "HEAD")
    branch = result.stdout.strip()

    if branch == "main":
        print("Cannot create PR from main branch. Switch to a feature branch first.")
        sys.exit(1)

    # Push first
    run_git("push", "-u", "origin", branch, check=False)

    # Create PR
    pr_body = f"## Summary\n{body}\n\n## Test plan\n- [ ] Manual testing\n- [ ] Unit tests pass"
    result = run_gh("pr", "create", "--title", title, "--body", pr_body)
    print(f"Pull request created: {result.stdout.strip()}")


def show_status():
    """Show repository status in a readable format."""
    # Current branch
    result = run_git("rev-parse", "--abbrev-ref", "HEAD")
    branch = result.stdout.strip()

    # Status
    status = run_git("status", "--short")

    # Recent commits
    log = run_git("log", "--oneline", "-5")

    print(f"Branch: {branch}")
    print(f"\nRecent commits:\n{log.stdout.strip()}")
    if status.stdout.strip():
        print(f"\nUncommitted changes:\n{status.stdout.strip()}")
    else:
        print("\nWorking directory clean.")


def show_log(count: int = 10):
    """Show recent commit history."""
    result = run_git("log", f"-{count}", "--oneline", "--graph", "--all")
    print(result.stdout.strip())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "init" and len(sys.argv) >= 3:
        init_project(sys.argv[2])
    elif command == "feature" and len(sys.argv) >= 3:
        create_feature_branch(sys.argv[2])
    elif command == "commit" and len(sys.argv) >= 4:
        conventional_commit(sys.argv[2], " ".join(sys.argv[3:]))
    elif command == "push":
        push_branch()
    elif command == "pr" and len(sys.argv) >= 4:
        create_pr(sys.argv[2], " ".join(sys.argv[3:]))
    elif command == "status":
        show_status()
    elif command == "log":
        count = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
        show_log(count)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)