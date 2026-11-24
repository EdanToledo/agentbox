#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -Eeuo pipefail

readonly RELEASE_BASE_URL="https://github.com/apptainer/apptainer/releases/download"

die() {
    echo "Error: $*" >&2
    exit 1
}

[[ -n "${APPTAINER_VERSION:-}" ]] || die "APPTAINER_VERSION is required"
[[ -n "${APPTAINER_SHA256:-}" ]] || die "APPTAINER_SHA256 is required"

[[ "$APPTAINER_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
    die "Invalid Apptainer version: $APPTAINER_VERSION"
[[ "$APPTAINER_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    die "Invalid Apptainer SHA-256 digest: $APPTAINER_SHA256"

for command in apt-get curl dpkg dpkg-deb mktemp mv sha256sum sudo uname; do
    command -v "$command" >/dev/null 2>&1 || die "Required command not found: $command"
done

[[ "$(uname -s)" == "Linux" ]] || die "Apptainer setup is supported only on Linux"
[[ "$(dpkg --print-architecture)" == "amd64" ]] || \
    die "Apptainer setup currently supports only the amd64 Debian architecture"

readonly filename="apptainer_${APPTAINER_VERSION}_amd64.deb"
readonly download_url="${RELEASE_BASE_URL}/v${APPTAINER_VERSION}/${filename}"
readonly temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"

download_dir="$(mktemp -d "${temp_root%/}/agentbox-apptainer.XXXXXX")"
readonly download_dir
trap 'rm -rf -- "$download_dir"' EXIT

readonly partial_path="${download_dir}/${filename}.partial"
readonly package_path="${download_dir}/${filename}"

echo "Downloading Apptainer ${APPTAINER_VERSION} for amd64"
if ! curl \
    --fail \
    --location \
    --proto '=https' \
    --proto-redir '=https' \
    --show-error \
    --silent \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 5 \
    --retry-max-time 900 \
    --connect-timeout 30 \
    --max-time 300 \
    --output "$partial_path" \
    "$download_url"; then
    die "Failed to download ${download_url} after retries"
fi
mv -- "$partial_path" "$package_path"

echo "Verifying ${filename}"
if ! printf '%s  %s\n' "$APPTAINER_SHA256" "$package_path" | sha256sum --check --status -; then
    die "Checksum mismatch for ${filename}"
fi

package_name="$(dpkg-deb --field "$package_path" Package)"
package_version="$(dpkg-deb --field "$package_path" Version)"
package_architecture="$(dpkg-deb --field "$package_path" Architecture)"
readonly package_name package_version package_architecture

[[ "$package_name" == "apptainer" ]] || die "Unexpected Debian package name: $package_name"
[[ "$package_version" == "$APPTAINER_VERSION" ]] || \
    die "Unexpected Debian package version: $package_version"
[[ "$package_architecture" == "amd64" ]] || \
    die "Unexpected Debian package architecture: $package_architecture"

echo "Updating APT package indexes"
if ! sudo env DEBIAN_FRONTEND=noninteractive \
    apt-get -o Acquire::Retries=3 update; then
    die "Failed to update APT package indexes"
fi

echo "Installing ${filename}"
if ! sudo env DEBIAN_FRONTEND=noninteractive \
    apt-get -o Acquire::Retries=3 install -y "$package_path"; then
    die "Failed to install ${filename}"
fi

command -v apptainer >/dev/null 2>&1 || die "Apptainer executable was not installed on PATH"
if ! installed_version="$(apptainer version)"; then
    die "Unable to determine the installed Apptainer version"
fi
readonly installed_version
[[ "$installed_version" == "$APPTAINER_VERSION" ]] || \
    die "Installed Apptainer version '$installed_version' does not match '$APPTAINER_VERSION'"

echo "Installed Apptainer ${installed_version}"
