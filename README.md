# speech2avatar snapshots

This repository is a snapshot collection of several `speech2avatar` / IMTalker-related working trees.

There is only one Git repository here at the top level:

- `speech2avatar-au-split-runpod-workflow`
- `speech2avatar-imf-complete`
- `speech2avatar-imf-original-backbone-minimal`
- `speech2avatar-moshi`
- `speech2avatar-static_vid`

There are no nested `.git` directories inside those subfolders, so this directory is safe to push as a normal single Git repository.

## Notes

- Most runnable code uses relative `./checkpoints` paths.
- RunPod-specific shell scripts may still default to `/workspace`, which is intentional for that environment.
- Large runtime outputs like checkpoints, experiment logs, previews, and cached Python files are ignored at the top level.

## Recommended branch to start from

If you are looking for the current minimal one-step iMF experiment with the restored original IMTalker-style backbone, start here:

- `speech2avatar-imf-original-backbone-minimal`

