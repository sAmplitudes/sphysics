#!/usr/bin/env bash
# Release a new version of sphysics.
#
# Tags the main branch of the GitLab repository with the new version, pushes the tag to GitLab,
# and pushes main and only this tag to the public GitHub repository. Other branches, commits,
# and tags are never pushed to GitHub.
#
# Environment variables:
#   SPHYSICS_GITLAB_REMOTE  git remote of the GitLab repository (default: origin)
#   SPHYSICS_GITHUB_URL     URL of the public GitHub repository
#                           (default: git@github.com:sAmplitudes/sphysics.git)

set -euo pipefail

readonly GITLAB_REMOTE="${SPHYSICS_GITLAB_REMOTE:-origin}"
readonly GITHUB_URL="${SPHYSICS_GITHUB_URL:-git@github.com:sAmplitudes/sphysics.git}"
# same version format as required by the deploy jobs in .gitlab-ci.yml
readonly VERSION_REGEX='^[0-9]+\.[0-9]+\.[0-9]+$'

usage() {
	cat <<EOF
Usage: $(basename "$0") [--dry-run] [X.Y.Z]

Release a new version of sphysics from the main branch of the GitLab repository.

  X.Y.Z      version to release (default: the latest version with its patch number increased by one)
  --dry-run  run all checks and show what would be done, but do not create or push anything
  -h, --help show this help

The local main branch must be identical to the main branch on GitLab. The script tags main with
the version, pushes the tag to GitLab, and pushes main and only this tag to the public GitHub
repository (${GITHUB_URL}).
EOF
}

die() {
	printf 'ERROR: %s\n' "$1" >&2
	shift
	(($# == 0)) || printf '       %s\n' "$@" >&2
	exit 1
}

info() {
	printf '>>> %s\n' "$*"
}

# Print the refs of a 'git ls-remote' listing that must not exist on GitHub, i.e., everything except
# HEAD, main, version tags, and the pull-request refs managed by GitHub.
unexpected_github_refs() {
	awk '$2 != "HEAD" && $2 != "refs/heads/main" && $2 !~ /^refs\/tags\/[0-9]+\.[0-9]+\.[0-9]+(\^\{\})?$/ && $2 !~ /^refs\/pull\// {print $2}' <<<"$1"
}

dry_run=false
version=""
for arg in "$@"; do
	case "${arg}" in
	--dry-run) dry_run=true ;;
	-h | --help)
		usage
		exit 0
		;;
	-*) die "Unknown option '${arg}'." ;;
	*)
		[[ -z "${version}" ]] || die "Only one version can be given."
		version="${arg}"
		;;
	esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
git remote get-url "${GITLAB_REMOTE}" >/dev/null 2>&1 || die "There is no git remote '${GITLAB_REMOTE}' (set SPHYSICS_GITLAB_REMOTE)."
gitlab_url="$(git remote get-url "${GITLAB_REMOTE}")"

# --- main branch on GitLab (fetched without tags, so that no private tags are fetched)
info "Fetching main from GitLab (${gitlab_url})"
git fetch --quiet --no-tags "${GITLAB_REMOTE}" "+refs/heads/main:refs/remotes/${GITLAB_REMOTE}/main"
main_commit="$(git rev-parse "refs/remotes/${GITLAB_REMOTE}/main")"
local_main="$(git rev-parse --verify --quiet refs/heads/main)" || die "There is no local branch main."
[[ "${local_main}" == "${main_commit}" ]] || die "The local main (${local_main:0:9}) differs from main on GitLab (${main_commit:0:9})." \
	"Merge and push all changes on GitLab and update the local main, e.g., with 'git switch main && git merge --ff-only ${GITLAB_REMOTE}/main'."

# --- public history on GitHub
info "Reading the refs of GitHub (${GITHUB_URL})"
github_refs="$(git ls-remote "${GITHUB_URL}")" || die "Cannot read the GitHub repository ${GITHUB_URL}."
unexpected="$(unexpected_github_refs "${github_refs}")"
[[ -z "${unexpected}" ]] || die "The GitHub repository contains unexpected refs:" "${unexpected}"
github_main="$(awk '$2 == "refs/heads/main" {print $1}' <<<"${github_refs}")"
[[ -n "${github_main}" ]] || die "The GitHub repository has no main branch."
git cat-file -e "${github_main}^{commit}" 2>/dev/null || die "main on GitHub (${github_main:0:9}) is not part of main on GitLab."
git merge-base --is-ancestor "${github_main}" "${main_commit}" || die "main on GitHub (${github_main:0:9}) is not an ancestor of main on GitLab; pushing would not be a fast-forward."
# a second root commit would mean that private history (e.g., an old branch) was merged into main
n_roots="$(git rev-list --max-parents=0 "${main_commit}" | wc -l)"
[[ "${n_roots}" -eq 1 ]] || die "main has ${n_roots} root commits instead of 1, i.e., it contains history that is not public. Nothing is pushed."

# --- version
github_versions="$(awk '$2 ~ /^refs\/tags\/[0-9]+\.[0-9]+\.[0-9]+$/ {sub("refs/tags/", "", $2); print $2}' <<<"${github_refs}")"
local_versions="$(git tag --merged "${main_commit}" | grep -E "${VERSION_REGEX}" || true)"
latest="$(printf '%s\n%s\n' "${github_versions}" "${local_versions}" | grep -E "${VERSION_REGEX}" | sort -V | tail -n 1 || true)"
if [[ -z "${version}" ]]; then
	[[ -n "${latest}" ]] || die "No previous version found; give the version explicitly."
	IFS=. read -r major minor patch <<<"${latest}"
	version="${major}.${minor}.$((patch + 1))"
fi
[[ "${version}" =~ ${VERSION_REGEX} ]] || die "Invalid version '${version}'; the format must be X.Y.Z."
if [[ -n "${latest}" ]]; then
	[[ "${version}" != "${latest}" && "$(printf '%s\n%s\n' "${latest}" "${version}" | sort -V | tail -n 1)" == "${version}" ]] ||
		die "Version ${version} is not higher than the latest version ${latest}."
fi
git rev-parse --verify --quiet "refs/tags/${version}" >/dev/null && die "Tag ${version} already exists locally."
gitlab_tag="$(git ls-remote --tags "${GITLAB_REMOTE}" "refs/tags/${version}")" || die "Cannot read the tags of the GitLab repository."
[[ -z "${gitlab_tag}" ]] || die "Tag ${version} already exists on GitLab."
! grep -q -E "[[:space:]]refs/tags/${version//./\\.}(\^\{\})?$" <<<"${github_refs}" || die "Tag ${version} already exists on GitHub."
tagged="$({
	git tag --points-at "${main_commit}"
	awk -v c="${main_commit}" '$1 == c && $2 ~ /^refs\/tags\// {sub("refs/tags/", "", $2); print $2}' <<<"${github_refs}"
} | grep -E "${VERSION_REGEX}" | sort -u -V | tr '\n' ' ' || true)"
[[ -z "${tagged}" ]] || die "main (${main_commit:0:9}) is already released as version ${tagged% }."

# --- summary and confirmation
n_new="$(git rev-list --count "${github_main}..${main_commit}")"
echo
echo "Release sphysics ${version} (latest version: ${latest:-none})"
echo "  commit:  $(git log -1 --format='%h %s' "${main_commit}")"
echo "  GitLab:  push tag ${version} to ${gitlab_url}"
echo "  GitHub:  push main and tag ${version} to ${GITHUB_URL}"
echo "  commits that become public on GitHub (${n_new}):"
git log --format='    %h %s' "${github_main}..${main_commit}"
echo
if [[ "${dry_run}" == true ]]; then
	info "Dry run: no tag was created and nothing was pushed."
	exit 0
fi
read -r -p "Proceed? [y/N] " answer || true
[[ "${answer:-}" =~ ^[yY]$ ]] || die "Aborted; nothing was created or pushed."

# --- tag and push
git tag "${version}" "${main_commit}"
info "Created tag ${version} on ${main_commit:0:9}"
if ! git push "${GITLAB_REMOTE}" "refs/tags/${version}:refs/tags/${version}"; then
	git tag -d "${version}" >/dev/null
	die "Pushing tag ${version} to GitLab failed; the local tag was deleted again."
fi
info "Pushed tag ${version} to GitLab"
retry="git push --atomic ${GITHUB_URL} ${main_commit}:refs/heads/main refs/tags/${version}:refs/tags/${version}"
git push --atomic "${GITHUB_URL}" "${main_commit}:refs/heads/main" "refs/tags/${version}:refs/tags/${version}" ||
	die "Pushing to GitHub failed; tag ${version} is already on GitLab. Retry with:" "  ${retry}"
info "Pushed main and tag ${version} to GitHub"

# --- check the result on GitHub
github_refs="$(git ls-remote "${GITHUB_URL}")" || die "Cannot read the GitHub repository ${GITHUB_URL} to check the result."
[[ "$(awk '$2 == "refs/heads/main" {print $1}' <<<"${github_refs}")" == "${main_commit}" ]] || die "main on GitHub is not ${main_commit:0:9} after the push."
[[ "$(awk -v t="refs/tags/${version}" '$2 == t {print $1}' <<<"${github_refs}")" == "${main_commit}" ]] || die "Tag ${version} on GitHub is not ${main_commit:0:9} after the push."
unexpected="$(unexpected_github_refs "${github_refs}")"
[[ -z "${unexpected}" ]] || die "The GitHub repository contains unexpected refs:" "${unexpected}"

github_web="https://github.com/$(sed -E 's#^(git@github\.com:|https://github\.com/)##; s#\.git$##' <<<"${GITHUB_URL}")"
echo
info "Released sphysics ${version}."
echo "The tag pipeline on GitLab deploys the documentation and the package."
echo "To archive the version on Zenodo, create a GitHub release for tag ${version}:"
echo "  ${github_web}/releases/new?tag=${version}"
