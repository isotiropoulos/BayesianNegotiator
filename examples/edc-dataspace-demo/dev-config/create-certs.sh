#!/usr/bin/env bash
# Generate the PKCS12 keystore each connector needs.
#
#   ./dev-config/create-certs.sh provider
#   ./dev-config/create-certs.sh consumer
#
# Self-signed and password-protected with a well-known password: this is a
# local demo, not a deployment. Do not reuse these certs anywhere real.
set -euo pipefail

ROLE="${1:?usage: create-certs.sh <provider|consumer>}"
PASSWORD="${KEYSTORE_PASSWORD:-datacellar}"
KEY_ALIAS="${KEY_ALIAS:-datacellar}"
OUT="$(cd "$(dirname "$0")" && pwd)/certs-${ROLE}"

if [[ -f "${OUT}/cert.pfx" ]]; then
  echo "keystore already present at ${OUT}/cert.pfx - skipping"
  exit 0
fi

mkdir -p "${OUT}"

openssl req -newkey rsa:2048 -nodes -x509 -days 365 \
  -keyout "${OUT}/key.pem" -out "${OUT}/cert.pem" \
  -subj "/CN=example-${ROLE}/O=BAMBON demo/C=EU" 2>/dev/null

openssl pkcs12 -export \
  -in "${OUT}/cert.pem" -inkey "${OUT}/key.pem" \
  -out "${OUT}/cert.pfx" -name "${KEY_ALIAS}" \
  -passout "pass:${PASSWORD}"

# The connector's vault reads the public certificate from this properties file
# under the `publickey` alias referenced in *.properties.
{
  echo "publickey=$(awk 'BEGIN{ORS="\\n"} {print}' "${OUT}/cert.pem")"
} > "${OUT}/vault.properties"

chmod 600 "${OUT}/key.pem" "${OUT}/cert.pfx"
echo "wrote ${OUT}/{cert.pem,key.pem,cert.pfx,vault.properties}"
