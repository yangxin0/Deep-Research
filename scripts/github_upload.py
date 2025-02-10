import os
import subprocess
import argparse
from pathlib import Path

def run_command(command, cwd=None):
    """Execute a shell command and return the output"""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, 
                              capture_output=True, cwd=cwd)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def setup_github_repo(repo_name, repo_url, project_path, branch_name="main", is_new_repo=True):
    """Setup and push to GitHub repository"""
    
    print(f"\n{'=' * 50}")
    print(f"Starting GitHub repository setup for: {repo_name}")
    print(f"{'=' * 50}\n")

    # Change to project directory
    os.chdir(project_path)
    
    if is_new_repo:
        print("Creating new repository...")
        
        # Initialize README if it doesn't exist
        readme_path = Path(project_path) / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w") as f:
                f.write(f"# {repo_name}\n")
            print("Created README.md")

        # Initialize git repository
        run_command("git init")
        print("Initialized git repository")

        # Add all files
        run_command("git add .")
        print("Added all files to git")

        # Initial commit
        run_command('git commit -m "Initial commit"')
        print("Created initial commit")

        # Set branch
        run_command(f"git branch -M {branch_name}")
        print(f"Set {branch_name} as current branch")

        # Add remote
        run_command(f"git remote add origin {repo_url}")
        print("Added remote origin")

    else:
        print("Setting up existing repository...")
        
        # Check if git is already initialized
        if not Path(project_path / ".git").exists():
            run_command("git init")
            print("Initialized git repository")

        # Create and switch to new branch if it's not main
        if branch_name != "main":
            run_command(f"git checkout -b {branch_name}")
            print(f"Created and switched to branch: {branch_name}")
        else:
            # Set main branch
            run_command("git branch -M main")
            print("Set main as default branch")

        # Add remote if not already added
        remotes = run_command("git remote -v")
        if not remotes or "origin" not in remotes:
            run_command(f"git remote add origin {repo_url}")
            print("Added remote origin")

        # Add all files
        run_command("git add .")
        print("Added all files to git")

        # Commit if there are changes
        status = run_command("git status")
        if "nothing to commit" not in status:
            run_command('git commit -m "Update repository"')
            print("Created commit")

    # Push to GitHub
    run_command(f"git push -u origin {branch_name}")
    print("\nSuccessfully pushed to GitHub!")
    print(f"Repository URL: {repo_url}")
    print(f"Branch: {branch_name}")

def main():
    parser = argparse.ArgumentParser(description="Upload project to GitHub repository")
    parser.add_argument("--repo-name", required=True, help="Name of the repository")
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL")
    parser.add_argument("--project-path", default=".", help="Path to the project directory")
    parser.add_argument("--new-repo", action="store_true", help="Create new repository")
    parser.add_argument("--branch", default="main", help="Branch name to create/push to")
    
    args = parser.parse_args()
    
    # Convert project path to absolute path
    project_path = Path(args.project_path).resolve()
    
    # Verify project path exists
    if not project_path.exists():
        print(f"Error: Project path {project_path} does not exist!")
        return

    setup_github_repo(args.repo_name, args.repo_url, project_path, args.branch, args.new_repo)

if __name__ == "__main__":
    main() 