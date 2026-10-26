#!/usr/bin/env bash
# One command for the whole story. See README.md for what each step shows.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> generating keystores"
./dev-config/create-certs.sh provider
./dev-config/create-certs.sh consumer

echo
echo "==> starting the data space (2 connectors, 2 BAMBONs, broker, backend)"
docker compose up -d --build --wait

# Build the one-shot runner images NOW, before seeding. `docker compose run
# --build` would rebuild them later and recreate the containers they depend on
# - including provider-connector, whose store is in memory, which silently
# throws away everything the seed just created.
docker compose build runner runner-sidechannel

echo
echo "==> seeding the provider with its restrictive published offer"
docker compose run --rm --no-deps -e EDC_CONNECTOR_HOST=provider-connector -e EDC_CONNECTOR_MANAGEMENT_PORT=19193 -e PROVIDER_BACKEND_URL=http://provider-backend:8080/data provider-negotiator python -m bambon_edc.seed_provider

echo
echo "==> negotiating between the two connectors, then pulling the dataset"
docker compose run --rm --no-deps runner

echo
echo "==> done. 'docker compose down -v' to clean up."
echo "    to re-run just the flow:   docker compose run --rm --no-deps runner"
echo "    to re-seed the provider:   see the seeding step above"
echo "    the HTTP side-channel variant, for comparison:"
echo "        docker compose run --rm --no-deps runner-sidechannel"
