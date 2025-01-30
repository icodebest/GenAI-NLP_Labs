import os

def auto_push(commit_message="Auto-update from Colab"):
    repo_dir = "/content/drive/MyDrive/MyProjectFolder"
    os.chdir(repo_dir)
    os.system("git add .")
    os.system(f'git commit -m "{commit_message}"')
    os.system("git push origin main")
    print("Changes pushed to GitHub successfully.")
