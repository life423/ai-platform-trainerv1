# Code Unification Plan

## Issue
The project currently has duplicate code in two locations:
1. `ai_platform_trainer/` - Main package directory
2. `src/` - Secondary source directory with similar structure

## Solution
We've decided to unify the code by keeping the `ai_platform_trainer/` directory as the single source of truth and removing the duplicate `src/` directory.

## Implementation Notes

1. The `src/` directory contains:
   - `src/ai_platform_trainer/` - A duplicate of the main package
   - `src/ai-platform-trainer/` - Another variant with slightly different structure

2. We attempted to remove the `src/` directory automatically, but encountered permission issues with Git objects.

3. **Manual Steps Required:**
   - Backup any unique code in `src/` that doesn't exist in `ai_platform_trainer/`
   - Delete the `src/` directory manually
   - Update import statements if necessary

4. Configuration has been consolidated:
   - Merged `settings.json` into `config.json`
   - Removed `settings.json` to avoid configuration duplication

## Verification
After removing the `src/` directory, ensure:
1. All tests pass
2. The game runs correctly
3. All AI training functionality works as expected