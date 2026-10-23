# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared network-safety helpers.

Today one control: :mod:`coa_common.net.url_guard`, which a caller-supplied URL
must pass before being fetched server-side. It lives in ``libs/common`` because
both halves of the Trino source type need it — discovery re-validates on every
scan, and the serve executor connects to the same stored URI.

``ontology-engine``'s remote-ontology fetch keeps its own equivalent
(``_resolve_and_validate_url``); consolidating the two is a separate change.
Its address-validation matrix lives in a *copy* of the logic
(``test_dns_rebinding_ssrf.py``), so only the non-disclosure properties are
tested against the real function.
"""

from .url_guard import UrlGuardError, ValidatedUrl, validate_url

__all__ = ["UrlGuardError", "ValidatedUrl", "validate_url"]
