#!/usr/bin/env bash
# Commita os artefatos de data/ de volta na branch do run.
#   tools/commit_artifacts.sh "mensagem do commit"
# JSON inválido (truncado, marcador de conflito) aborta antes do commit.
# Push com pull --rebase e até 3 tentativas; se não subir, o step falha.
set -euo pipefail

msg="$1"
branch="${GITHUB_REF_NAME:?GITHUB_REF_NAME não definido}"

git config user.name  "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

git add -A data/history
git add -A data/*.json 2>/dev/null || true

if git diff --cached --quiet; then
  echo "Sem artefatos novos para commitar."
  exit 0
fi

git diff --cached --name-only --diff-filter=AM -- '*.json' '*.jsonl' |
while read -r f; do
  python - "$f" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    if path.endswith(".jsonl"):
        for n, line in enumerate(fh, 1):
            if line.strip():
                json.loads(line)
    else:
        json.load(fh)
PY
done

git commit -m "$msg"

for attempt in 1 2 3; do
  if git pull --rebase origin "$branch" && git push origin "HEAD:$branch"; then
    exit 0
  fi
  git rebase --abort 2>/dev/null || true
  echo "Push falhou (tentativa $attempt de 3)."
  sleep $((attempt * 10))
done

echo "::error::Artefatos não subiram depois de 3 tentativas."
exit 1
