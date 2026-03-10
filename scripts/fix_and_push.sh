#!/usr/bin/env bash
# Fix and push awesome-generative-ai-guide
# Usage: ./scripts/fix_and_push.sh [fork|origin]
#   fork   - push to your fork (default)
#   origin - push to upstream (requires write access)

set -e
cd "$(dirname "$0")/.."

REMOTE="${1:-origin}"
echo "=== awesome-generative-ai-guide: fix and push to $REMOTE ==="

git checkout main 2>/dev/null || true

# Commit if there are changes
if [[ -n $(git status -s) ]]; then
  git add -A
  git commit -m "fix: correct repo URLs, rename AI evals chapter

- Replace awesome-generative-ai-resources with awesome-generative-ai-guide (broken image links)
- Rename 01_wth_are_ai_evals.md to 01_what_are_ai_evals.md
- Update Cite Us journal URL in README"
  echo "✓ Commit created"
else
  echo "No changes to commit"
fi

# Push
echo ""
echo "Pushing to $REMOTE/main..."
if git push "$REMOTE" main 2>/dev/null; then
  echo "✓ Push successful!"
  [[ "$REMOTE" == "origin" ]] && echo "Open PR: https://github.com/aishwaryanr/awesome-generative-ai-guide/compare"
else
  echo "✗ Push failed. If email privacy error: use GitHub no-reply email in git config"
  echo "  git config user.email 'USERNAME@users.noreply.github.com'"
  exit 1
fi
