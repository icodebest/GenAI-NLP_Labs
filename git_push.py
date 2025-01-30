import os

def auto_push(commit_message="Auto-update from Colab"):
    repo_dir = "/content/drive/MyDrive/MyProjectFolder"
    os.chdir(repo_dir)

    # Fix authentication issue
    os.system('git remote set-url origin https://icodebest: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@github.com/icodebest/GenAI-NLP_Labs.git')

    # Add, commit, and push
    os.system("git add .")
    os.system(f'git commit -m "{commit_message}"')
    os.system("git push origin main")  # Change 'main' if necessary

    print("Changes pushed to GitHub successfully.")
