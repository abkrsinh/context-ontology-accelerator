# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a single Trino statement and return all of its rows.

Deliberately the same shape as :class:`~.athena_statement.AthenaStatementRunner`
so the two can be read side by side: ``run(statement, *, catalog, schema,
timeout_seconds) -> list[list[str | None]]``, every failure raising rather than
returning a short answer. The two rules that module holds to hold here for the
same reason — a statement that half-succeeds would ship a table stripped of its
columns, and enrichment would then invent descriptions over the gap.

Three things are simpler than the Athena path, and they are why this is ~300
lines rather than 600:

* **Trino has ONE parser**, so identifiers may be double-quoted everywhere.
  ``AthenaStatementRunner``'s callers must pass identifiers unquoted because
  Athena's two parsers disagree (``SHOW``/``DESCRIBE`` reject double quotes,
  ``SELECT`` rejects backticks). That constraint does not apply, so this module
  quotes consistently — see :func:`quote_ident`.
* **No header row, ever.** The client hands back data rows only; column names
  live on ``cursor.description``. There is no ``StatementType`` to interpret and
  no ``rows[1:]`` strip to get wrong.
* **No result pagination to hand-roll.** ``trino.dbapi`` follows the
  coordinator's ``nextUri`` chain internally, so ``fetchall()`` is already
  "paginate to exhaustion". This is the concrete reason the official client is a
  dependency rather than raw HTTP against ``/v1/statement``.

Retries are split across two layers that cover **disjoint** failure classes, so
unlike the Athena runner's botocore problem they cannot multiply:

* The **client** retries the HTTP layer — 429 (honouring ``Retry-After``), 502,
  503, 504, connection errors, read timeouts, and a 200 with an empty body. Its
  own policy is left on deliberately. Pinning ``max_attempts=1`` and
  reimplementing this here is not possible without parsing status codes out of
  exception *messages*: ``Http502Error``/``503``/``504`` are bare ``Exception``
  subclasses carrying no status attribute, and every other status — 429
  included — arrives as a generic ``HttpError`` whose only record of the code is
  the string ``"error 429: ..."``. The ``Retry-After`` header is not on the
  exception at all.
* This module retries what the client cannot see as retryable: a query that
  reached the coordinator and came back with an ``errorName`` naming a busy or
  restarting cluster. Those arrive as ``TrinoQueryError`` after a perfectly
  successful HTTP exchange, so no HTTP-layer policy will ever retry them.
"""

from __future__ import annotations

import base64
import json
import random
import time
from typing import Any
from urllib.parse import urlsplit

import structlog
import trino
from coa_common.aws_config import RetryPolicy
from trino import exceptions as trino_exc

from coa_sources.database.metrics import emit_metric

# structlog, not stdlib logging — see athena_catalog.py: the Lambda's
# setup_logging() pins the stdlib root logger to WARNING and omits ExtraAdder,
# so `extra={...}` on a stdlib logger is silently discarded.
logger = structlog.get_logger(__name__)

# Wall-clock budget for one statement, submission to last row, INCLUDING every
# retry at both layers. Lower than the Athena runner's 120 s because there is no
# customer Lambda cold start on this path — the coordinator is already running
# and `information_schema` is served from its metadata caches. Still generous
# enough for a cold Iceberg REST catalog to answer a whole-schema column read,
# and kept well under the discovery Lambda's timeout so one pathological
# statement cannot consume the pass.
DEFAULT_STATEMENT_TIMEOUT_S = 60.0

# Per-HTTP-request timeout handed to the client, distinct from the budget above.
# The client applies it to each request in the nextUri chain, so a long result
# set legitimately spans many of them; the total stays bounded by the budget.
_REQUEST_TIMEOUT_S = 15.0

# HTTP attempts the CLIENT makes per request. Trino's own default, named rather
# than left implicit because it is load-bearing twice: it is the layer that
# handles 429/50x and Retry-After (see the module docstring), and it multiplies
# the per-request timeout, so `_execute` divides the remaining budget by it.
_CLIENT_HTTP_ATTEMPTS = 3

# Attempts THIS module makes, for coordinator-reported cluster-busy errors only.
# Lower than the Athena runner's 5 because the HTTP layer below has already
# absorbed the transient-transport class; what is left is a cluster that needs
# time to free resources, and a fourth attempt inside one scan's budget does not
# meaningfully improve on three.
_RETRY_ATTEMPTS = 3

# Trino's own default. Named rather than inlined because it is the port assumed
# when `uri` carries none, and `https` is the only scheme this accepts.
_DEFAULT_HTTPS_PORT = 443

# Trino error names that describe the cluster being busy or briefly unavailable
# rather than the query being wrong. Matched on the `errorName` the coordinator
# returns, not on message text: the message carries operator- and
# connector-authored content, and a substring match on it would turn a real
# error into three pointless re-runs.
_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "NO_NODES_AVAILABLE",
        "REMOTE_TASK_ERROR",
        "SERVER_STARTING_UP",
        "SERVER_SHUTTING_DOWN",
        "QUERY_QUEUE_FULL",
        "CLUSTER_OUT_OF_MEMORY",
    }
)

# Fallback `user` when a JWT carries no readable `sub`. A constant, not a
# generated value: it appears in Trino's query history, where a stable name is
# what makes COA's traffic attributable.
_DEFAULT_USER = "coa-discovery"


class TrinoStatementError(RuntimeError):
    """Base for every way a discovery statement can fail to produce rows.

    Callers catch this to account for a failed source without distinguishing
    connection failures from query failures — but the subclasses stay distinct
    because "the coordinator refused this query" and "we never reached it" call
    for different operator action.
    """


class TrinoStatementFailed(TrinoStatementError):
    """The coordinator rejected or failed the query.

    ``error_name`` is Trino's own classification (e.g. ``TABLE_NOT_FOUND``,
    ``PERMISSION_DENIED``) and is preserved because it, not the message, is what
    distinguishes an operator-fixable grant problem from a bug here — and it is
    what :func:`_is_retryable` reads to decide whether a busy cluster deserves
    another attempt.
    """

    def __init__(self, statement: str, error_name: str, message: str) -> None:
        """Record Trino's classification of the failure alongside its message."""
        self.statement = statement
        self.error_name = error_name
        self.message = message
        detail = f" [{error_name}]" if error_name else ""
        super().__init__(f"Trino statement failed{detail}: {message} — {statement}")


class TrinoStatementTimeout(TrinoStatementError):
    """The statement did not complete within its total budget."""

    def __init__(self, statement: str, timeout_seconds: float) -> None:
        """Record the budget that was exhausted, for the operator-facing message."""
        self.statement = statement
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Trino statement did not complete within {timeout_seconds:g}s — {statement}")


class TrinoConnectionFailed(TrinoStatementError):
    """The coordinator was unreachable, or rejected our credential.

    Distinct from :class:`TrinoStatementFailed` because nothing was executed: at
    create time this is the difference between a 502 (the cluster is down, worth
    retrying) and a 400 (the operator's credential or URI is wrong).

    Not retried by this module — by the time one of these surfaces, the client's
    own HTTP policy has already exhausted ``_CLIENT_HTTP_ATTEMPTS``.
    """

    def __init__(self, statement: str, cause: Exception, *, auth_failed: bool = False) -> None:
        """Wrap ``cause``, keeping it reachable on both ``cause`` and ``__cause__``."""
        self.statement = statement
        self.cause = cause
        self.auth_failed = auth_failed
        what = "authentication rejected" if auth_failed else "connection failed"
        super().__init__(f"Trino {what}: {cause} — {statement}")


class TrinoConfigError(TrinoStatementError):
    """The source's stored configuration cannot be turned into a connection.

    Raised before any network call, so it is never confused with the cluster
    being unreachable. Covers a URI the model's pattern admits but this module
    cannot use, and a secret whose shape contradicts ``authMode``.
    """


def quote_literal(value: str) -> str:
    """Return ``value`` as a single-quoted Trino string literal.

    The literal-position counterpart to :func:`quote_ident`. Needed because a
    name that is legal as an identifier may still carry a ``'``, and a name is
    sometimes carried as data — a label in a result column, say — where it is no
    longer in an identifier position and doubling ``"`` protects nothing.
    """
    return "'" + value.replace("'", "''") + "'"


def quote_ident(name: str) -> str:
    """Return ``name`` as a double-quoted Trino identifier.

    Safe in every identifier position because Trino has one parser, unlike
    Athena. Interior double quotes are doubled per the SQL standard, which is
    what stops a crafted catalog or schema name from closing the quote and
    appending SQL. That matters even though `information_schema` predicates take
    bound parameters: the catalog and schema also arrive as *identifiers*, in
    positions where a parameter is not accepted.
    """
    return '"' + name.replace('"', '""') + '"'


def _split_uri(uri: str) -> tuple[str, int]:
    """Split a validated ``https`` coordinator URI into host and port.

    The model's ``TrinoUri`` pattern already excludes a non-``https`` scheme, a
    path, a query and a fragment, so this does not re-validate those — it raises
    only on what a pattern cannot catch, an unparseable authority. Port defaults
    to 443 rather than Trino's 8080: the scheme is pinned to ``https``, so a
    plaintext default would be a surprising fallback.
    """
    parts = urlsplit(uri)
    if parts.scheme != "https" or not parts.hostname:
        raise TrinoConfigError(f"Trino uri must be https with a host, got {uri!r}")
    try:
        port = parts.port or _DEFAULT_HTTPS_PORT
    except ValueError as exc:  # non-numeric port; urlsplit defers this to access
        raise TrinoConfigError(f"Trino uri has an unusable port: {uri!r}") from exc
    return parts.hostname, port


def _auth_from_secret(auth_mode: str, secret: dict[str, Any]) -> tuple[Any, str]:
    """Build the client's auth object and the Trino username from a secret.

    Returns ``(auth, user)``. Trino wants a ``user`` even when the credential is
    a bearer token — it is what appears in the query history and in
    ``current_user``, so it is worth being accurate rather than passing a
    constant.

    The shape is checked here as well as at create time, which is not redundant:
    a secret can be rotated into a contradicting shape after the source exists,
    and that must name the mismatch rather than surface as an opaque 401.
    """
    mode = (auth_mode or "").upper()
    if mode == "BASIC":
        username, password = secret.get("username"), secret.get("password")
        if not username or not password:
            raise TrinoConfigError("authMode BASIC requires a secret with 'username' and 'password'")
        if "token" in secret:
            raise TrinoConfigError("secret carries both BASIC and JWT keys; refusing to guess the mode")
        return trino.auth.BasicAuthentication(username, password), username
    if mode == "JWT":
        token = secret.get("token")
        if not token:
            raise TrinoConfigError("authMode JWT requires a secret with 'token'")
        if "username" in secret or "password" in secret:
            raise TrinoConfigError("secret carries both JWT and BASIC keys; refusing to guess the mode")
        # The JWT's own `sub` is the identity Trino enforces against, so the
        # username here is cosmetic — but Trino rejects a connection with none.
        return trino.auth.JWTAuthentication(token), _subject_of(token)
    # Reached only if TrinoAuthMode gains a member without this mapping growing
    # one. Loud rather than defaulted: a silent fallback to BASIC would send a
    # token as a password.
    raise TrinoConfigError(f"unsupported Trino authMode {auth_mode!r}")


def _subject_of(token: str) -> str:
    """Best-effort ``sub`` claim from a JWT, for Trino's ``user``.

    The payload is decoded but NOT verified — Trino verifies the token, and a
    signature check here would need the IdP's keys for no security gain. The
    value is used only for attribution, so a malformed token degrades to a
    constant rather than failing the connection: refusing to connect because a
    cosmetic field would not parse is the wrong trade.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # restore stripped base64url padding
        claims = json.loads(base64.urlsafe_b64decode(payload))
        subject = claims.get("sub")
        return str(subject) if subject else _DEFAULT_USER
    except Exception:  # noqa: BLE001 - a cosmetic field must not fail the connection
        return _DEFAULT_USER


def _is_retryable(exc: TrinoStatementError) -> bool:
    """Whether a failure is worth another attempt at THIS layer.

    Only one class qualifies: a query that reached the coordinator and came back
    with an ``errorName`` naming a busy or restarting cluster. Everything else is
    terminal here, for two different reasons:

    * **Transport and HTTP failures** (``TrinoConnectionFailed``) have already
      been retried ``_CLIENT_HTTP_ATTEMPTS`` times by the client, including 429
      with ``Retry-After``. Retrying again would multiply the two layers — 3 x 3
      — which is the trap ``athena_statement`` avoids by pinning botocore to 1.
    * **Query errors with any other ``errorName``** — a missing schema, a
      syntax error, a denied grant — fail identically every time, so retrying
      only delays an error the operator needs to see.

    A rejected credential is terminal too, and deliberately so: retrying it
    three times against a password authenticator risks an account lockout.
    """
    if not isinstance(exc, TrinoStatementFailed):
        return False
    return exc.error_name.upper() in _RETRYABLE_ERROR_NAMES


class TrinoStatementRunner:
    """Runs one Trino statement to completion and returns every row.

    One instance per source, built from that source's resolved configuration.
    Unlike :class:`~.athena_statement.AthenaStatementRunner` there is no
    long-lived client to share: ``trino.dbapi.connect`` is cheap and a connection
    carries per-statement state, so one is opened and closed per ``run``. That
    also keeps a rotated credential from being pinned for the life of the
    instance.
    """

    def __init__(
        self,
        *,
        uri: str,
        auth_mode: str,
        secret: dict[str, Any],
        timeout_seconds: float = DEFAULT_STATEMENT_TIMEOUT_S,
        request_timeout_s: float = _REQUEST_TIMEOUT_S,
        connect: Any | None = None,
    ) -> None:
        """Resolve the coordinator target and credential.

        Args:
            uri: Coordinator URI, ``https`` with optional port. Already matched
                against ``TrinoUri`` and passed the SSRF guard by the caller —
                this module re-runs neither, so it must not be handed a raw
                caller-supplied value.
            auth_mode: ``BASIC`` or ``JWT``, from ``TrinoConfiguration``.
            secret: Decoded credential secret; shape must match ``auth_mode``.
            timeout_seconds: Default TOTAL wall-clock budget per statement,
                shared across retries rather than reset by each.
            request_timeout_s: Per-HTTP-request timeout within that budget.
            connect: ``trino.dbapi.connect`` substitute, for tests.

        Raises:
            TrinoConfigError: the URI or the secret shape is unusable. Raised
                here, before any network call, so a configuration error is never
                reported as the cluster being down.
        """
        self._host, self._port = _split_uri(uri)
        self._auth, self._user = _auth_from_secret(auth_mode, secret)
        self._timeout_seconds = timeout_seconds
        self._request_timeout_s = request_timeout_s
        self._connect = connect or trino.dbapi.connect

    def run(
        self,
        statement: str,
        *,
        catalog: str | None = None,
        schema: str | None = None,
        params: tuple[Any, ...] | None = None,
        timeout_seconds: float | None = None,
    ) -> list[list[str | None]]:
        """Execute ``statement`` and return every result row.

        Args:
            statement: The SQL to run. Identifiers may be double-quoted — see
                the module docstring on Trino's single parser — and value
                predicates should be passed as ``params`` rather than
                interpolated.
            catalog: Session catalog, so an unqualified ``information_schema``
                resolves within it.
            schema: Session schema. This is the *session default*, not a
                predicate: a ``table_schema`` filter still has to appear in the
                SQL, because `information_schema` prunes on the predicate.
            params: Bound parameters for ``?`` placeholders in ``statement``.
            timeout_seconds: Overrides the instance default. This is the TOTAL
                budget: retries at both layers share it rather than each getting
                a fresh one.

        Returns:
            One list of cell values per row, in Trino's order. Cells are
            stringified to match ``AthenaStatementRunner``'s contract, so the two
            are interchangeable to a caller; ``None`` stays ``None`` and remains
            distinct from ``""``.

        Raises:
            TrinoStatementFailed: the coordinator rejected or failed the query.
            TrinoStatementTimeout: the total budget ran out.
            TrinoConnectionFailed: the coordinator was unreachable, or the
                credential was rejected.
        """
        budget = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + budget
        policy = RetryPolicy.for_async()
        last_error: TrinoStatementError | None = None

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return self._execute(statement, catalog, schema, params, deadline, budget)
            except TrinoStatementTimeout:
                emit_metric("TrinoStatementTimeout", 1, "Count")
                raise
            except TrinoStatementError as exc:
                if not _is_retryable(exc):
                    raise
                last_error = exc

            if attempt == _RETRY_ATTEMPTS:
                emit_metric("TrinoStatementRetriesExhausted", 1, "Count")
                raise last_error
            # Capped exponential backoff with full jitter, per the repo's async
            # retry policy. Full jitter rather than a fixed delay because a
            # queue-full or starting-up coordinator rejects every in-flight
            # statement at once, and a fixed delay would have them all return
            # together and be rejected again.
            delay = min(policy.base_delay * 2 ** (attempt - 1), policy.max_delay)
            sleep_for = random.uniform(0, delay)  # noqa: S311 - jitter, not crypto
            # Do not sleep past the budget only to fail on the next attempt.
            if time.monotonic() + sleep_for >= deadline:
                emit_metric("TrinoStatementTimeout", 1, "Count")
                raise TrinoStatementTimeout(statement, budget) from last_error
            logger.warning(
                "trino_statement_retrying",
                statement=statement,
                error_name=getattr(last_error, "error_name", ""),
                attempt=attempt,
                max_attempts=_RETRY_ATTEMPTS,
                sleep_seconds=round(sleep_for, 3),
            )
            time.sleep(sleep_for)

        # Unreachable: the loop either returns or raises at the final attempt.
        # Present so the function has no implicit None path.
        raise TrinoStatementTimeout(statement, budget)

    def _execute(
        self,
        statement: str,
        catalog: str | None,
        schema: str | None,
        params: tuple[Any, ...] | None,
        deadline: float,
        budget: float,
    ) -> list[list[str | None]]:
        """One attempt: connect, execute, drain, close.

        ``fetchall`` is already exhaustive — ``trino.dbapi`` follows the
        coordinator's ``nextUri`` chain internally — so there is no pagination
        loop here and no way to silently truncate, which is the failure the
        Athena runner needs 60 lines to prevent.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TrinoStatementTimeout(statement, budget)

        connection = None
        try:
            connection = self._connect(
                host=self._host,
                port=self._port,
                user=self._user,
                auth=self._auth,
                http_scheme="https",
                catalog=catalog,
                schema=schema,
                source="coa-discovery",
                max_attempts=_CLIENT_HTTP_ATTEMPTS,
                # Divided by the client's attempt count so its own retries fit
                # inside what is left of the budget rather than overrunning it,
                # and floored at 1 s so a nearly-exhausted budget still makes a
                # real request instead of a zero-timeout one that fails
                # instantly and burns an attempt.
                request_timeout=max(1.0, min(self._request_timeout_s, remaining / _CLIENT_HTTP_ATTEMPTS)),
            )
            cursor = connection.cursor()
            if params:
                cursor.execute(statement, params)
            else:
                cursor.execute(statement)
            rows = cursor.fetchall()
        except trino_exc.TrinoAuthError as exc:
            emit_metric("TrinoAuthFailed", 1, "Count")
            raise TrinoConnectionFailed(statement, exc, auth_failed=True) from exc
        except trino_exc.TrinoQueryError as exc:
            emit_metric("TrinoStatementFailed", 1, "Count", ErrorName=exc.error_name or "UNKNOWN")
            raise TrinoStatementFailed(statement, exc.error_name or "", str(exc.message or exc)) from exc
        except (trino_exc.TrinoConnectionError, trino_exc.HttpError, OSError) as exc:
            emit_metric("TrinoUnreachable", 1, "Count")
            raise TrinoConnectionFailed(statement, exc) from exc
        finally:
            if connection is not None:
                self._close_quietly(connection)

        return [[None if cell is None else str(cell) for cell in row] for row in rows]

    @staticmethod
    def _close_quietly(connection: Any) -> None:
        """Best-effort close; a failure here must not mask the real error.

        Debug, not warning: unlike an abandoned Athena execution — which keeps
        holding a slot in an account-wide DML quota — a leaked Trino connection
        costs the coordinator one idle HTTP client that its own timeouts reap.
        """
        try:
            connection.close()
        except Exception:  # noqa: BLE001 - the statement's own error is the one worth raising
            logger.debug("trino_connection_close_failed", exc_info=True)
