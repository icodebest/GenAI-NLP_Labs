import os

def auto_push(commit_message="Auto-update from Colab"):
    repo_dir = "/content/drive/MyDrive/MyProjectFolder"
    os.chdir(repo_dir)

    print("📌 Changing directory to repo:", os.getcwd())

    # Set authentication using environment variable
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("❌ ERROR: GitHub token not found! Run os.environ['GITHUB_TOKEN'] = 'your-token' before pushing.")
        return

    os.system(f'git remote set-url origin https://icodebest:{github_token}@github.com/icodebest/GenAI-NLP_Labs.git')

    # Add changes
    os.system("git add .")
    os.system(f'git commit -m "{commit_message}"')
    os.system("git push origin main")

    print("✅ Changes pushed to GitHub successfully.")
