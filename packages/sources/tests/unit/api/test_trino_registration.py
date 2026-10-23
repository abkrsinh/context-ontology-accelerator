# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRINO source registration.

Two properties carry the weight:

* **The SSRF guard runs before the record exists.** ``uri`` is caller-supplied and
  fetched server-side from a private subnet, so an unguarded create is an internal
  port scanner. The allowlist is fail-closed, and these tests assert the refusal
  rather than the acceptance.
* **The four caller-supplied identifiers are lowercased in the STORED blob.** Trino's
  SPI only ever reports lowercase, so a ``ORDERS`` filter silently matches nothing;
  normalising at read instead would leave the source detail page showing a filter
  that is not the one applied.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ALLOWED_ORIGIN", "*")

import coa_sources.api.sources_handler  # noqa: F401, I001, E402
import coa_sources.api.database_routes as _dr  # noqa: E402, I001
from coa_control_plane_server.models.query_engine import QueryEngine  # noqa: E402
from coa_control_plane_server.models.source_sub_type import SourceSubType  # noqa: E402
from tests.unit.conftest import dao_double  # noqa: E402

pytestmark = pytest.mark.unit

_DR = "coa_sources.api.database_routes"
_NAMESPACE_ID = "11111111-2222-3333-4444-555555555555"
_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:team/trino-AbCdEf"


def _config(**overrides) -> dict:
    base = {
        "uri": "https://trino.corp.com:8443",
        "catalog": "iceberg",
        "schemaName": "sales",
        "credentialSecretArn": _SECRET_ARN,
        "authMode": "JWT",
    }
    base.update(overrides)
    return base


def _make_trino_db_req(**overrides):
    """A CreateDatabaseSourceInput-shaped mock carrying only the Trino config.

    The other four members are ``None`` explicitly: a MagicMock auto-attribute is
    truthy, which would trip the mutual-exclusivity guard.
    """
    config = _config(**overrides)
    req = MagicMock()
    req.name = "prod-lakehouse"
    req.glue_configuration = None
    req.jdbc_configuration = None
    req.custom_connector_configuration = None
    req.databricks_sql_warehouse_configuration = None
    trino = MagicMock()
    trino.uri = config["uri"]
    trino.catalog = config["catalog"]
    trino.schema_name = config["schemaName"]
    trino.credential_secret_arn = config["credentialSecretArn"]
    trino.auth_mode = config["authMode"]
    trino.to_dict.return_value = config
    req.trino_configuration = trino
    req.metadata_enrichment_enabled = None
    return req


def _source_put(mock_dao) -> dict:
    puts = [c.args[0] for c in mock_dao.put.call_args_list if str(c.args[0].get("SK", "")).startswith("SRC#")]
    assert len(puts) == 1, f"expected one source put, got {len(puts)}"
    return puts[0]


def _create(req, *, allowed_hosts=("trino.corp.com",), binding_ok=True, retryable=False):
    """Run a create with DNS, the secret binding and the allowlist all stubbed."""
    dao = dao_double()
    verdict = MagicMock(ok=binding_ok, retryable=retryable, message="secret binding refused")
    with (
        patch(f"{_DR}._get_dao", return_value=dao),
        patch(f"{_DR}._get_scan_dao", return_value=dao_double()),
        patch(f"{_DR}._get_sqs", return_value=MagicMock()),
        patch(f"{_DR}.adjust_namespace_source_count"),
        patch(f"{_DR}.check_secret_namespace_binding", return_value=verdict),
        patch(f"{_DR}._TRINO_ALLOWED_HOSTS", tuple(allowed_hosts)),
        patch("coa_common.net.url_guard.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.5", 0))]),
    ):
        raw = _dr._create_database_source(req, _NAMESPACE_ID, namespace_language="en")
    return raw["statusCode"], json.loads(raw["body"]), dao


# ── The record ───────────────────────────────────────────────────────────────


def test_a_trino_config_derives_the_trino_sub_type() -> None:
    status, _body, dao = _create(_make_trino_db_req())
    assert status == 202
    assert _source_put(dao)["sourceSubType"] == SourceSubType.TRINO


def test_query_engine_is_trino_not_athena() -> None:
    """Athena cannot reach a Trino-only table, so there is no fallback to choose."""
    _status, _body, dao = _create(_make_trino_db_req())
    assert _source_put(dao)["queryEngine"] == QueryEngine.TRINO


def test_no_athena_catalog_is_claimed() -> None:
    """Nothing is provisioned for a Trino source, so there is no name to release."""
    _status, _body, dao = _create(_make_trino_db_req())
    item = _source_put(dao)
    assert not item.get("athenaDataCatalogName")
    assert not item.get("glueConnectionName")


# ── Identifier normalisation ─────────────────────────────────────────────────


def test_the_stored_blob_lowercases_all_four_identifiers() -> None:
    req = _make_trino_db_req(
        catalog="Iceberg",
        schemaName="SALES",
        tableFilter="ORDERS|Customers",
        tableExcludeFilter="STAGING_*",
    )
    _status, _body, dao = _create(req)
    blob = json.loads(_source_put(dao)["configuration"])
    assert blob["catalog"] == "iceberg"
    assert blob["schemaName"] == "sales"
    assert blob["tableFilter"] == "orders|customers"
    assert blob["tableExcludeFilter"] == "staging_*"


def test_absent_filters_are_not_invented() -> None:
    _status, _body, dao = _create(_make_trino_db_req())
    blob = json.loads(_source_put(dao)["configuration"])
    assert "tableFilter" not in blob
    assert "tableExcludeFilter" not in blob


def test_the_uri_is_not_lowercased() -> None:
    """Only identifiers fold; a path-less host is already case-insensitive in DNS."""
    _status, _body, dao = _create(_make_trino_db_req(uri="https://Trino.Corp.com:8443"))
    blob = json.loads(_source_put(dao)["configuration"])
    assert blob["uri"] == "https://Trino.Corp.com:8443"


# ── SC-1: the SSRF guard ─────────────────────────────────────────────────────


def test_a_private_address_is_refused_when_the_host_is_not_allowlisted() -> None:
    status, body, dao = _create(_make_trino_db_req(), allowed_hosts=("other.corp.com",))
    assert status == 400
    assert "trinoConfiguration.uri" in body["error"]
    assert dao.put.call_args_list == []


def test_an_empty_allowlist_blocks_creation_outright() -> None:
    """Fail-closed: the allowlist is the only thing distinguishing the operator's
    in-VPC coordinator from an arbitrary internal host."""
    status, _body, dao = _create(_make_trino_db_req(), allowed_hosts=())
    assert status == 400
    assert dao.put.call_args_list == []


def test_a_non_https_uri_is_refused() -> None:
    status, body, _dao = _create(_make_trino_db_req(uri="http://trino.corp.com"))
    assert status == 400
    assert "https" in body["error"]


def test_the_imds_address_is_refused_even_when_the_host_is_allowlisted() -> None:
    """The hard-deny classes are not purchasable with an allowlist entry."""
    dao = dao_double()
    with (
        patch(f"{_DR}._get_dao", return_value=dao),
        patch(f"{_DR}._get_scan_dao", return_value=dao_double()),
        patch(f"{_DR}._get_sqs", return_value=MagicMock()),
        patch(f"{_DR}.adjust_namespace_source_count"),
        patch(f"{_DR}.check_secret_namespace_binding", return_value=MagicMock(ok=True)),
        patch(f"{_DR}._TRINO_ALLOWED_HOSTS", ("trino.corp.com",)),
        patch(
            "coa_common.net.url_guard.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("169.254.169.254", 0))],
        ),
    ):
        raw = _dr._create_database_source(_make_trino_db_req(), _NAMESPACE_ID, namespace_language="en")
    assert raw["statusCode"] == 400
    assert dao.put.call_args_list == []


def test_the_error_does_not_echo_the_resolved_address() -> None:
    status, body, _dao = _create(_make_trino_db_req(), allowed_hosts=())
    assert status == 400
    assert "10.0.0.5" not in body["error"]


# ── Secret binding ───────────────────────────────────────────────────────────


def test_a_secret_not_bound_to_this_namespace_is_refused() -> None:
    status, _body, dao = _create(_make_trino_db_req(), binding_ok=False)
    assert status == 400
    assert dao.put.call_args_list == []


def test_a_retryable_binding_failure_is_a_503_not_a_400() -> None:
    """So the caller retries rather than believing the ARN was rejected."""
    status, _body, _dao = _create(_make_trino_db_req(), binding_ok=False, retryable=True)
    assert status == 503


# ── No effect on existing sub-types ──────────────────────────────────────────


def _make_glue_db_req():
    """A minimal Glue request double; every other config member explicitly None."""
    req = MagicMock()
    req.name = "my-glue-db"
    req.jdbc_configuration = None
    req.custom_connector_configuration = None
    req.databricks_sql_warehouse_configuration = None
    req.trino_configuration = None
    req.metadata_enrichment_enabled = None
    glue = MagicMock()
    glue.database_name = "mydb"
    glue.region = "us-east-1"
    glue.catalog_id = "123456789012"
    glue.athena_data_catalog_name = None
    glue.execution_engine = None
    glue.redshift_workgroup_name = None
    glue.to_dict.return_value = {"databaseName": "mydb", "region": "us-east-1"}
    req.glue_configuration = glue
    return req


def test_the_guard_and_normalisation_do_not_run_for_a_non_trino_source() -> None:
    """A Glue request must not touch the allowlist, DNS, or the lowercasing."""
    dao = dao_double()
    with (
        patch(f"{_DR}._get_dao", return_value=dao),
        patch(f"{_DR}._get_scan_dao", return_value=dao_double()),
        patch(f"{_DR}._get_sqs", return_value=MagicMock()),
        patch(f"{_DR}.adjust_namespace_source_count"),
        patch(f"{_DR}._TRINO_ALLOWED_HOSTS", ()),
        patch("coa_common.net.url_guard.socket.getaddrinfo", side_effect=AssertionError("DNS must not be used")),
    ):
        raw = _dr._create_database_source(_make_glue_db_req(), _NAMESPACE_ID, namespace_language="en")
    assert raw["statusCode"] == 202
    assert _source_put(dao)["queryEngine"] == QueryEngine.ATHENA
