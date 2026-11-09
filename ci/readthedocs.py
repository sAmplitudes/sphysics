#!/usr/bin/env python3
"""Publish the documentation built by the GitLab CI on Read the Docs.

Read the Docs (RTD) does not build the documentation itself (see .readthedocs.yaml). Instead, it
serves the HTML that the build_pages job of the GitLab CI built for the same commit, i.e., the
files that the pages job deploys to GitLab Pages. This script has two modes:

fetch    Runs on RTD. Looks up the successful GitLab pipeline of the commit that RTD is building,
         downloads the artifacts of its build_pages job, and extracts the public/ directory to
         $READTHEDOCS_OUTPUT/html. If the GitLab pipeline has not finished yet (RTD starts a build
         as soon as a tag is pushed to GitHub, while the GitLab pipeline is still running), the
         script exits with code 183, which makes RTD cancel the build instead of failing it; the
         readthedocs job of the GitLab CI triggers a new RTD build once the artifacts exist.

trigger  Runs in the readthedocs job of the GitLab CI (tag pipelines only). Triggers the RTD builds
         of the released version, of latest (main on GitHub, which is the same commit because
         scripts/release.sh pushes main and the tag together), and of stable (RTD's alias of the
         highest version tag).

Environment variables:
  GITLAB_TOKEN       fetch: GitLab token with the read_api scope for the project (a project access
                     token with the Reporter role is sufficient). Set it in the RTD project settings
                     under "Environment variables"; do not expose it to pull request builds.
  READTHEDOCS_TOKEN  trigger: RTD API token (https://app.readthedocs.org/accounts/tokens/), stored
                     as a masked CI/CD variable of the GitLab project.
  READTHEDOCS_GIT_COMMIT_HASH, READTHEDOCS_OUTPUT  set by RTD
  CI_COMMIT_TAG                                    set by the GitLab CI
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

GITLAB_URL = os.environ.get("SPHYSICS_GITLAB_URL", "https://gitlab.desy.de")
GITLAB_PROJECT = os.environ.get("SPHYSICS_GITLAB_PROJECT", "belle2/physics/amplitudesatbelleii/sphysics")
DOCS_JOB = os.environ.get("SPHYSICS_DOCS_JOB", "build_pages")
DOCS_DIR = os.environ.get("SPHYSICS_DOCS_DIR", "public")  # directory in the artifacts of DOCS_JOB
READTHEDOCS_URL = "https://app.readthedocs.org"
READTHEDOCS_PROJECT = "sphysics"
# version tags, same format as in the rules of .gitlab-ci.yml and in scripts/release.sh
VERSION_TAG = re.compile(r"\d+\.\d+\.\d+")
# RTD cancels the build (instead of marking it as failed) if a command exits with this code
RTD_CANCEL_BUILD = 183


def info(message: str) -> None:
    print(f">>> {message}", flush=True)


def die(message: str, exit_code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(exit_code)


def request(url: str, headers: dict[str, str], method: str = "GET") -> urllib.request.addinfourl:
    req = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=300)


# --- fetch ------------------------------------------------------------------------------------


def gitlab_api(path: str, query: dict[str, str] | None = None) -> str:
    """Return the URL of a GitLab API endpoint of the project."""
    url = f"{GITLAB_URL}/api/v4/projects/{urllib.parse.quote(GITLAB_PROJECT, safe='')}/{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return url


def gitlab_get(path: str, token: str | None, query: dict[str, str] | None = None):
    headers = {"PRIVATE-TOKEN": token} if token else {}
    try:
        with request(gitlab_api(path, query), headers) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        if err.code in (401, 404):
            die(
                f"GitLab API returned {err.code} for {GITLAB_URL}/{GITLAB_PROJECT} ({path}). "
                "The project is private: GITLAB_TOKEN must be a valid token with the read_api scope "
                "for the project, set as environment variable of the Read the Docs project."
            )
        raise


def fetch() -> None:
    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        info("GITLAB_TOKEN is not set; accessing GitLab without authentication")
    commit = os.environ["READTHEDOCS_GIT_COMMIT_HASH"]
    output = Path(os.environ["READTHEDOCS_OUTPUT"]) / "html"

    # One commit can have several pipelines (push to main, merge request, tag). Prefer the
    # pipeline of the version tag, i.e., the one that deploys GitLab Pages; it also matters for
    # the version shown in the documentation, which docs/conf.py takes from the Git tags at build
    # time. Otherwise, use the newest pipeline (the API lists them newest first).
    pipelines = gitlab_get("pipelines", token, {"sha": commit, "status": "success", "per_page": "100"})
    pipelines.sort(key=lambda p: not VERSION_TAG.fullmatch(p["ref"] or ""))
    for pipeline in pipelines:
        jobs = gitlab_get(f"pipelines/{pipeline['id']}/jobs", token, {"scope[]": "success", "per_page": "100"})
        job = next((j for j in jobs if j["name"] == DOCS_JOB and j.get("artifacts_file")), None)
        if job is not None:
            break
    else:
        print(
            f"No successful pipeline of {GITLAB_URL}/{GITLAB_PROJECT} for commit {commit} has a "
            f"successful {DOCS_JOB} job with artifacts (pipelines checked: {len(pipelines)}).\n"
            "Either the GitLab pipeline is still running, in which case its readthedocs job triggers "
            "a new Read the Docs build when the documentation is built, or the artifacts expired.\n"
            f"Cancelling this build (exit code {RTD_CANCEL_BUILD})."
        )
        sys.exit(RTD_CANCEL_BUILD)
    info(f"Pipeline {pipeline['web_url']} ({pipeline['ref']}), job {job['web_url']}")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "artifacts.zip"
        headers = {"PRIVATE-TOKEN": token} if token else {}
        with request(gitlab_api(f"jobs/{job['id']}/artifacts"), headers) as resp, archive.open("wb") as f:
            shutil.copyfileobj(resp, f)
        info(f"Downloaded the artifacts ({archive.stat().st_size / 1e6:.1f} MB)")
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.startswith(f"{DOCS_DIR}/")]
            if not members:
                die(f"The artifacts of job {job['web_url']} do not contain the directory {DOCS_DIR}/.")
            zf.extractall(tmp, members)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp) / DOCS_DIR, output, dirs_exist_ok=True)
    n_files = sum(1 for p in output.rglob("*") if p.is_file())
    info(f"Copied {n_files} files to {output}")
    if not (output / "index.html").is_file():
        die(f"{output} contains no index.html")


# --- trigger ----------------------------------------------------------------------------------


def trigger() -> None:
    token = os.environ.get("READTHEDOCS_TOKEN") or die("READTHEDOCS_TOKEN is not set (masked CI/CD variable).")
    tag = os.environ.get("CI_COMMIT_TAG") or die("CI_COMMIT_TAG is not set; the trigger runs only in tag pipelines.")
    headers = {"Authorization": f"Token {token}"}
    failed = False
    for version in (tag, "latest", "stable"):
        url = f"{READTHEDOCS_URL}/api/v3/projects/{READTHEDOCS_PROJECT}/versions/{version}/builds/"
        try:
            with request(url, headers, method="POST") as resp:
                build = json.load(resp).get("build", {})
                info(f"Triggered the Read the Docs build of version {version}: {build.get('urls', {}).get('build', '')}")
        except urllib.error.HTTPError as err:
            failed = True
            body = err.read().decode(errors="replace")[:500]
            print(f"ERROR: Triggering the build of version {version} failed with HTTP {err.code}: {body}", file=sys.stderr)
            if err.code == 404:
                print(
                    f"       Version {version} is unknown or inactive on Read the Docs. New tags are activated by the "
                    "automation rule of the project (see .readthedocs.yaml); check the versions at "
                    f"{READTHEDOCS_URL}/projects/{READTHEDOCS_PROJECT}/versions/",
                    file=sys.stderr,
                )
    if failed:
        sys.exit(1)


def main() -> None:
    modes = {"fetch": fetch, "trigger": trigger}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        die(f"Usage: {sys.argv[0]} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
