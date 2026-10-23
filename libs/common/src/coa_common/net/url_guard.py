# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSRF guard for a caller-supplied URL fetched server-side.

Without this, accepting a URL from an API caller and fetching it from inside a
private subnet turns source creation into an internal port scanner. The guard
resolves the host ONCE, validates every address it resolved to, and returns a
URL pinned to a validated address with the original hostname for the ``Host``
header — closing the DNS-rebinding TOCTOU window where an attacker-controlled
nameserver answers differently for the check and for the fetch.

Two gates, both of which must pass:

* **Hard deny**, regardless of any allowlist: loopback, link-local (which covers
  IMDS at 169.254.169.254), unspecified, multicast and reserved.
* **Reachability**, which depends on ``require_global``:
    - ``True`` — the address must be globally routable. This is right for
      fetching public content.
    - ``False`` — a non-globally-routable address is allowed ONLY if the
      hostname is on ``allowed_hosts``. A Trino coordinator is plausibly on a
      private address (in-VPC, peered, PrivateLink), and requiring public
      routability would push operators to expose it to the internet: worse
      security dressed as stricter.

An empty ``allowed_hosts`` with ``require_global=False`` therefore admits
nothing that is not already public — fail-closed.

Two subtleties worth stating, because both are easy to get wrong:

* Reachability is gated on ``not is_global``, not on ``is_private``. CGNAT
  (100.64.0.0/10) is neither private nor any hard-deny class, yet it is not
  globally routable — an ``is_private`` test would let it through.
* IPv4-mapped IPv6 needs no special handling: ``ipaddress`` delegates the flags
  of ``::ffff:169.254.169.254`` to the embedded IPv4, so it reports
  ``is_link_local`` and is hard-denied like the bare form.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["UrlGuardError", "ValidatedUrl", "validate_url"]

_ALLOWED_SCHEMES = frozenset({"https"})
_DEFAULT_PORT = 443


class UrlGuardError(ValueError):
    """A URL was refused. Framework-neutral so every caller can adapt it.

    Deliberately a ``ValueError`` and not an HTTP exception: this module is
    imported by a Lambda handler, a FastAPI route and a discovery worker, and
    coupling it to one framework is what kept the original copy unshareable.
    """


@dataclass(frozen=True)
class ValidatedUrl:
    """A URL proven safe to fetch, pinned to the address that was validated.

    Attributes:
        pinned_url: Connect to THIS, not the original. Its host is the validated
            IP, so no second DNS lookup can redirect the request.
        hostname: The original hostname. Send it as ``Host``, and as the TLS
            server name — connecting to a bare IP otherwise fails certificate
            verification against any SNI-based host.
        port: The resolved port, defaulted to 443.
    """

    pinned_url: str
    hostname: str
    port: int


def _host_is_allowed(hostname: str, allowed_hosts: Sequence[str]) -> bool:
    """Whether ``hostname`` matches the allowlist exactly or as a domain suffix.

    Suffix matching requires a label boundary, so an entry of ``corp.com``
    admits ``trino.corp.com`` but NOT ``evil-corp.com`` — a plain ``endswith``
    would admit both, which is the classic allowlist bypass.
    """
    host = hostname.rstrip(".").casefold()
    for entry in allowed_hosts:
        candidate = entry.strip().rstrip(".").casefold().lstrip("*.")
        if candidate and (host == candidate or host.endswith(f".{candidate}")):
            return True
    return False


def validate_url(
    url: str,
    *,
    require_global: bool = False,
    allowed_hosts: Sequence[str] = (),
    field_name: str = "url",
) -> ValidatedUrl:
    """Validate ``url`` and return it pinned to a verified address.

    Args:
        url: The caller-supplied URL. Must be ``https``.
        require_global: When true, every resolved address must be globally
            routable and ``allowed_hosts`` is not consulted.
        allowed_hosts: Hostnames or domain suffixes whose non-public addresses
            are acceptable. Ignored when ``require_global`` is true; an empty
            sequence otherwise means nothing non-public is admitted.
        field_name: Named in error messages, so the operator learns which input
            was refused.

    Raises:
        UrlGuardError: wrong scheme, no hostname, a non-ASCII hostname,
            unresolvable, or any resolved address refused by either gate.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise UrlGuardError(f"{field_name} must use https://")
    if not parsed.hostname:
        raise UrlGuardError(f"{field_name} is missing a hostname")
    # A raw-unicode hostname is refused rather than IDNA-encoded, so a homograph
    # cannot be normalised into something that matches an allowlist entry.
    # Explicit punycode (`xn--`) is left to the gates below, where the allowlist
    # is what decides.
    if not parsed.hostname.isascii():
        raise UrlGuardError(f"{field_name} hostname must be ASCII; supply punycode explicitly")

    hostname = parsed.hostname
    try:
        # Resolved ONCE. Every later use is of a validated address, which is
        # what removes the rebinding window.
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UrlGuardError(f"Cannot resolve {field_name} host") from exc
    if not infos:
        raise UrlGuardError(f"{field_name} hostname resolved to no addresses")

    allowlisted = (not require_global) and _host_is_allowed(hostname, allowed_hosts)
    validated: list[str] = []
    # EVERY address, not the first: a host answering with one public and one
    # loopback address would otherwise pass validation and could be fetched at
    # either, depending on resolver ordering.
    for *_unused, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        # Never echo the address. Reporting what a hostname resolved to would
        # make this check an internal-DNS oracle.
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip.is_reserved:
            raise UrlGuardError(f"{field_name} resolves to a forbidden address")
        if not ip.is_global and not allowlisted:
            raise UrlGuardError(f"{field_name} resolves to a non-routable address and its host is not allowlisted")
        validated.append(str(ip))

    pinned_ip = validated[0]
    port = parsed.port or _DEFAULT_PORT
    netloc = f"[{pinned_ip}]:{port}" if ":" in pinned_ip else f"{pinned_ip}:{port}"
    return ValidatedUrl(pinned_url=parsed._replace(netloc=netloc).geturl(), hostname=hostname, port=port)
