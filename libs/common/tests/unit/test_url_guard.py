# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SSRF guard.

Every rejection is a vector this guard exists to close, so the tests are written
as the attack rather than as the branch: IMDS by both spellings, a split answer
where only one address is bad, rebinding, and an allowlist that must not match on
a substring.

DNS is faked throughout — ``getaddrinfo`` is the only external dependency, and
pinning it is what makes "what if the second lookup differs" testable at all.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest
from coa_common.net.url_guard import UrlGuardError, validate_url

pytestmark = pytest.mark.unit

_MODULE = "coa_common.net.url_guard"


def _resolves_to(*addresses: str):
    """Patch ``getaddrinfo`` to answer with ``addresses``, in order."""
    infos = [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addresses]
    return patch(f"{_MODULE}.socket.getaddrinfo", return_value=infos)


# ── Scheme and hostname ──────────────────────────────────────────────────────


@pytest.mark.parametrize("url", ["http://trino.example.com", "file:///etc/passwd", "ftp://x.example.com", "//x.com"])
def test_rejects_non_https(url: str) -> None:
    with pytest.raises(UrlGuardError, match="https"):
        validate_url(url, require_global=True)


def test_rejects_missing_hostname() -> None:
    with pytest.raises(UrlGuardError, match="hostname"):
        validate_url("https://", require_global=True)


def test_rejects_non_ascii_hostname() -> None:
    """A raw-unicode homograph is refused rather than IDNA-normalised."""
    with pytest.raises(UrlGuardError, match="ASCII"):
        validate_url("https://trıno.example.com", require_global=True)


def test_punycode_is_left_to_the_gates() -> None:
    """Explicit punycode is not refused outright; the allowlist decides."""
    with _resolves_to("93.184.216.34"):
        assert validate_url("https://xn--trno-5qa.example.com", require_global=True).hostname.startswith("xn--")


# ── Hard-deny classes ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",  # IMDS
        "::ffff:169.254.169.254",  # IMDS, IPv4-mapped IPv6
        "127.0.0.1",
        "::1",
        "::ffff:127.0.0.1",
        "0.0.0.0",
        "::",
        "224.0.0.1",  # multicast — is_global is True, so needs its own check
        "240.0.0.1",  # reserved
        "fe80::1",  # IPv6 link-local
    ],
)
def test_hard_denied_addresses_are_refused_even_when_allowlisted(address: str) -> None:
    """The allowlist cannot buy reachability to these — that is what makes it hard."""
    with _resolves_to(address), pytest.raises(UrlGuardError, match="forbidden"):
        validate_url("https://trino.corp.com", allowed_hosts=["trino.corp.com"])


def test_the_error_never_echoes_the_resolved_address() -> None:
    """Reporting it would make the guard an internal-DNS oracle."""
    with _resolves_to("10.1.2.3"), pytest.raises(UrlGuardError) as exc:
        validate_url("https://trino.example.com")
    assert "10.1.2.3" not in str(exc.value)


# ── require_global mode ──────────────────────────────────────────────────────


def test_public_address_passes_in_global_mode() -> None:
    with _resolves_to("93.184.216.34"):
        result = validate_url("https://example.com", require_global=True)
    assert result.hostname == "example.com"
    assert result.port == 443


@pytest.mark.parametrize("address", ["10.0.0.5", "192.168.1.1", "172.16.0.1", "100.64.0.1"])
def test_global_mode_refuses_non_public_even_when_allowlisted(address: str) -> None:
    with _resolves_to(address), pytest.raises(UrlGuardError, match="non-routable"):
        validate_url("https://trino.corp.com", require_global=True, allowed_hosts=["trino.corp.com"])


# ── Allowlist mode ───────────────────────────────────────────────────────────


def test_private_address_rejected_when_not_allowlisted() -> None:
    """Fail-closed: an empty allowlist admits nothing non-public."""
    with _resolves_to("10.0.0.5"), pytest.raises(UrlGuardError, match="allowlisted"):
        validate_url("https://trino.corp.com", allowed_hosts=[])


def test_private_address_allowed_when_host_allowlisted() -> None:
    """The in-VPC Trino coordinator — the topology this mode exists for."""
    with _resolves_to("10.0.0.5"):
        assert validate_url("https://trino.corp.com", allowed_hosts=["trino.corp.com"]).hostname == "trino.corp.com"


def test_cgnat_needs_the_allowlist_too() -> None:
    """100.64.0.0/10 is neither private nor hard-denied, but is not routable.

    Gating on ``is_private`` instead of ``not is_global`` would admit it.
    """
    with _resolves_to("100.64.0.1"), pytest.raises(UrlGuardError, match="allowlisted"):
        validate_url("https://trino.corp.com", allowed_hosts=[])
    with _resolves_to("100.64.0.1"):
        assert validate_url("https://trino.corp.com", allowed_hosts=["trino.corp.com"])


def test_domain_suffix_entry_admits_a_subdomain() -> None:
    with _resolves_to("10.0.0.5"):
        assert validate_url("https://trino.corp.com", allowed_hosts=["corp.com"])


def test_allowlist_suffix_is_not_a_substring_match() -> None:
    """`evil-corp.com` must not satisfy an entry of `corp.com`."""
    with _resolves_to("10.0.0.5"), pytest.raises(UrlGuardError, match="allowlisted"):
        validate_url("https://evil-corp.com", allowed_hosts=["corp.com"])


def test_allowlist_matching_is_case_and_trailing_dot_insensitive() -> None:
    with _resolves_to("10.0.0.5"):
        assert validate_url("https://TRINO.Corp.COM.", allowed_hosts=["  corp.com  "])


def test_wildcard_prefix_in_an_entry_is_tolerated() -> None:
    """Operators write `*.corp.com`; treat it as the suffix they meant."""
    with _resolves_to("10.0.0.5"):
        assert validate_url("https://trino.corp.com", allowed_hosts=["*.corp.com"])


def test_an_allowlisted_host_still_gets_a_public_address_accepted() -> None:
    with _resolves_to("93.184.216.34"):
        assert validate_url("https://trino.corp.com", allowed_hosts=["trino.corp.com"])


# ── Multiple addresses and rebinding ─────────────────────────────────────────


def test_rejects_when_any_resolved_address_is_denied() -> None:
    """A split answer must fail: resolver ordering would otherwise decide safety."""
    with _resolves_to("93.184.216.34", "127.0.0.1"), pytest.raises(UrlGuardError, match="forbidden"):
        validate_url("https://example.com", require_global=True)


def test_rejects_when_the_denied_address_comes_first() -> None:
    with _resolves_to("127.0.0.1", "93.184.216.34"), pytest.raises(UrlGuardError, match="forbidden"):
        validate_url("https://example.com", require_global=True)


def test_resolution_happens_once() -> None:
    """Every address is checked from a single lookup, so a split answer cannot be
    evaluated twice with different results."""
    with _resolves_to("93.184.216.34") as getaddrinfo:
        validate_url("https://example.com", require_global=True)
    assert getaddrinfo.call_count == 1


# ── Resolution failures ──────────────────────────────────────────────────────


def test_unresolvable_host_is_refused() -> None:
    with (
        patch(f"{_MODULE}.socket.getaddrinfo", side_effect=socket.gaierror("nodename nor servname")),
        pytest.raises(UrlGuardError, match="Cannot resolve"),
    ):
        validate_url("https://nope.example.com", require_global=True)


def test_resolution_error_does_not_leak_the_resolver_message() -> None:
    with (
        patch(f"{_MODULE}.socket.getaddrinfo", side_effect=socket.gaierror("nodename nor servname")),
        pytest.raises(UrlGuardError) as exc,
    ):
        validate_url("https://nope.example.com", require_global=True)
    assert "nodename" not in str(exc.value)


def test_empty_resolution_is_refused() -> None:
    with (
        patch(f"{_MODULE}.socket.getaddrinfo", return_value=[]),
        pytest.raises(UrlGuardError, match="no addresses"),
    ):
        validate_url("https://example.com", require_global=True)


# ── Port and IPv6 handling ───────────────────────────────────────────────────


def test_explicit_port_is_preserved() -> None:
    with _resolves_to("93.184.216.34"):
        assert validate_url("https://example.com:8443", require_global=True).port == 8443


def test_port_defaults_to_443() -> None:
    with _resolves_to("93.184.216.34"):
        assert validate_url("https://example.com", require_global=True).port == 443


def test_an_ipv6_only_host_is_accepted() -> None:
    with _resolves_to("2606:2800:220:1:248:1893:25c8:1946"):
        assert validate_url("https://example.com", require_global=True).hostname == "example.com"


def test_field_name_appears_in_the_error() -> None:
    """So the operator learns which input was refused, not just that one was."""
    with pytest.raises(UrlGuardError, match="trinoConfiguration.uri"):
        validate_url("http://x.example.com", field_name="trinoConfiguration.uri")
