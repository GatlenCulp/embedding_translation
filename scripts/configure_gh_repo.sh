#!/usr/bin/env bash

# ./configure_gh_repo.sh [--no-github] <dir> <repo_name> <protections>

# Ex: ./scripts/configure_gh_repo.sh --no-github /Users/hugz/Work-Projects/project-boilerplate/tmp/gatlens_new_project my-repo main_and_dev
# Ex: ./scripts/configure_gh_repo.sh /Users/hugz/Work-Projects/project-boilerplate/tmp/gatlens_new_project my-repo main_and_dev

set -eo pipefail

# Function to display usage information
usage() {
    echo "Usage: $0 [--no-github] <dir> <repo_name> <protections>"
    echo "  --no-github: Optional. If set, skips GitHub operations and only sets up the local repository."
    echo "  dir: Directory where the repository will be created or updated"
    echo "  repo_name: Name of the repository"
    echo "  protections: 'none' (unprotect main), 'main' (protected main), or 'main_and_dev' (protected main, unprotected dev)"
    exit 1
}

# Initialize variables
NO_GITHUB=false

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-github) NO_GITHUB=true; shift ;;
        *) break ;;
    esac
done

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    usage
fi

# Assign arguments to variables
DIR=$1
REPO_NAME=$2
PROTECTIONS=$3

# Validate directory
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

# Validate repository name
if [[ ! "$REPO_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: Invalid repository name. Use only letters, numbers, underscores, and hyphens."
    usage
fi

# Validate protection level
if [[ ! "$PROTECTIONS" =~ ^(none|main|main_and_dev)$ ]]; then
    echo "Error: Invalid protection level. Use 'none', 'main', or 'main_and_dev'."
    usage
fi

# Check for gh CLI only if --no-github is not set
if [ "$NO_GITHUB" = false ]; then
    # Confirm gh CLI is installed
    if ! command -v gh &> /dev/null; then
        echo "Error: gh CLI is required to run this script with GitHub operations."
        echo "Use --no-github to skip GitHub operations or install gh CLI."
        exit 1
    fi

    # Confirm gh CLI is authenticated
    if ! gh auth status &> /dev/null; then
        echo "Error: gh CLI is not authenticated. Please run 'gh auth login'."
        echo "Use --no-github to skip GitHub operations or authenticate gh CLI."
        exit 1
    fi
fi

# Change to the specified directory
cd "$DIR" || exit 1

# Initialize repository if it doesn't exist
if [ ! -d .git ]; then
    git init
fi

# Add all files and commit if there are changes
if [ -n "$(git status --porcelain)" ]; then
    git add .
    git commit -m "Initial commit"
fi

# Add semantic versioning tag if it doesn't exist
if ! git rev-parse v0.1.0 >/dev/null 2>&1; then
    git tag -a v0.1.0 -m "Initial version"
fi

# Create dev branch if requested and it doesn't exist
if [ "$PROTECTIONS" = "main_and_dev" ]; then
    if ! git rev-parse --verify dev >/dev/null 2>&1; then
        git branch dev
    fi
fi

# GitHub operations
if [ "$NO_GITHUB" = false ]; then
    # Get the authenticated user's username
    GITHUB_USERNAME=$(gh api user -q .login)

    # Create repository on GitHub using gh CLI if it doesn't exist
    if ! gh repo view "$GITHUB_USERNAME/$REPO_NAME" &>/dev/null; then
        gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
    else
        # If repo exists, ensure origin is set correctly
        git remote set-url origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git" || git remote add origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
    fi

    # Push main branch and tags
    git push -u origin main
    git push --tags

    # Push dev branch if it exists
    if git rev-parse --verify dev >/dev/null 2>&1; then
        git push -u origin dev
    fi

    # Check if the repository is public
    IS_PUBLIC=$(gh api "repos/$GITHUB_USERNAME/$REPO_NAME" -q .private | grep -q false && echo true || echo false)

    if [ "$IS_PUBLIC" = "true" ]; then
        # Set branch protections using gh CLI
        if [ "$PROTECTIONS" = "main" ] || [ "$PROTECTIONS" = "main_and_dev" ]; then
            gh api "repos/$GITHUB_USERNAME/$REPO_NAME/branches/main/protection" \
                -X PUT \
                -H "Accept: application/vnd.github.v3+json" \
                -F required_status_checks='{"strict": true, "contexts": []}' \
                -F enforce_admins=true \
                -F required_pull_request_reviews='{"required_approving_review_count":1}' \
                -F restrictions=null
        fi

        if [ "$PROTECTIONS" = "main_and_dev" ]; then
            gh api "repos/$GITHUB_USERNAME/$REPO_NAME/branches/dev/protection" \
                -X PUT \
                -H "Accept: application/vnd.github.v3+json" \
                -F required_status_checks='{"strict": true, "contexts": []}' \
                -F enforce_admins=true \
                -F required_pull_request_reviews='{"required_approving_review_count":1}' \
                -F restrictions=null
        fi
        echo "Branch protections set successfully."
    else
        echo "Warning: Branch protections can only be set for public repositories or with a GitHub Pro account."
        echo "The repository is private, so branch protections were not set."
    fi

    echo "Repository configuration complete on GitHub!"
else
    echo "Local repository configuration complete!"
fi