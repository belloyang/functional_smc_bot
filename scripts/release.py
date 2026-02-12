#!/usr/bin/env python3
import re
import sys
import subprocess
import argparse
from pathlib import Path

VERSION_FILE = Path("app/__init__.py")

def run_command(command, dry_run=False):
    print(f"Executing: {' '.join(command)}")
    if not dry_run:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result.stdout
    return ""

def get_current_version():
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        print("Could not find __version__ in app/__init__.py")
        sys.exit(1)
    return match.group(1)

def update_version_file(new_version, dry_run=False):
    print(f"Updating app/__init__.py to version {new_version}")
    if not dry_run:
        content = VERSION_FILE.read_text()
        new_content = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{new_version}"', content)
        VERSION_FILE.write_text(new_content)

def bump_version(version, bump_type):
    # Remove suffix if present
    base_version = version.split('-')[0]
    parts = list(map(int, base_version.split('.')))
    
    if bump_type == 'patch':
        parts[2] += 1
    elif bump_type == 'minor':
        parts[1] += 1
        parts[2] = 0
    elif bump_type == 'major':
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
        
    return ".".join(map(str, parts))

def main():
    parser = argparse.ArgumentParser(description="Automate release and version bumping.")
    parser.add_argument("--type", choices=["patch", "minor", "major"], default="patch", help="Type of version bump.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes.")
    args = parser.parse_args()

    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # 1. Release Phase: Remove -rc suffix
    release_version = current_version.split('-')[0]
    if release_version == current_version:
        print(f"Version {current_version} is already a release version. Skipping release phase.")
    else:
        update_version_file(release_version, args.dry_run)
        run_command(["git", "add", str(VERSION_FILE)], args.dry_run)
        run_command(["git", "commit", "-m", f"Release version {release_version}"], args.dry_run)
        run_command(["git", "tag", "-a", f"v{release_version}", "-m", f"Version {release_version}"], args.dry_run)
        print(f"✅ Released and tagged v{release_version}")

    # 2. Bump Phase: Increment version and add -rc
    next_base_version = bump_version(release_version, args.type)
    next_version = f"{next_base_version}-rc"
    
    update_version_file(next_version, args.dry_run)
    run_command(["git", "add", str(VERSION_FILE)], args.dry_run)
    run_command(["git", "commit", "-m", f"Bump version to {next_version}"], args.dry_run)
    
    print(f"✅ Version bumped to {next_version}")
    print("\nNext steps: git push && git push --tags")

if __name__ == "__main__":
    main()
