# ✦ Context Ontology Accelerator

An open-source semantic context layer for AWS that combines knowledge graphs, formal ontologies, and rule-based systems with modern AI — enabling agents to retrieve context, validate it against business logic, and determine correct actions.

> **Branch `trino-source`: work in progress, based on `v0.3.7`.**
>
> Adds Trino as a data source (`SourceSubType.TRINO`), so a lakehouse reached through Trino (Iceberg REST, Hive metastore, Nessie, Delta) can be onboarded without mirroring into Glue or operating an Athena federation Lambda. Discovery reads the coordinator's `information_schema` in two queries per source; serve-time SQL executes on that same coordinator, so discovered identifiers are the queryable ones.
>
> **Landed so far**
>
> - [`models/src/main/smithy/unified-sources.smithy`](models/src/main/smithy/unified-sources.smithy) — `TrinoConfiguration` and `TrinoAuthMode`; `TRINO` added to `SourceSubType` and `QueryEngine`; a `trinoConfiguration` member on both `CreateDatabaseSourceInput` and `DatabaseSourceDetail`. Run `make generate` after pulling.
> - [`connectors/trino_statement.py`](packages/sources/src/coa_sources/database/connectors/trino_statement.py) — `TrinoStatementRunner`: runs one statement to completion and returns every row, raising rather than returning a short answer. Deliberately shaped like its `athena_statement.py` sibling.
> - [`connectors/trino_source.py`](packages/sources/src/coa_sources/database/connectors/trino_source.py) — `TrinoConnector`: discovery in **two** queries per source, no per-table fan-out. Resolves the schema's canonical spelling from `information_schema.schemata` first, because a mis-cased predicate makes Trino drop it and return zero rows with no error. Selects `comment` explicitly (a hidden column, absent from `SELECT *`), maps `is_nullable` to `Column.nullable` with `@notnull` still able to tighten it, and preserves `ordinal_position` order because composite primary keys depend on it.
> - [`connectors/__init__.py`](packages/sources/src/coa_sources/database/connectors/__init__.py) — registry entry mapping `SourceSubType.TRINO` to `TrinoConnector`.
> - [`pipeline/federation_handler.py`](packages/sources/src/coa_sources/database/pipeline/federation_handler.py) — explicit skip: nothing to provision and no Lake Formation grant to make.
> - [`database/sub_types.py`](packages/sources/src/coa_sources/database/sub_types.py) — new `NO_CATALOG_TEARDOWN_SUB_TYPES`. A Trino source provisions no catalog, and both existing teardown paths would act on resources that never existed and report success.
> - [`libs/common/.../constants.py`](libs/common/src/coa_common/constants.py), [`api/sources_handler.py`](packages/sources/src/coa_sources/api/sources_handler.py), [`coa_serve/clients/athena.py`](packages/context-manager/src/coa_serve/clients/athena.py) — the shared sub-type mirror, the config-blob map, and a serve log label. A new `SourceSubType` member has six forced registration sites; five are guarded by tripwire tests that fail the moment the member exists.
> - Tests: [`test_trino_statement.py`](packages/sources/tests/unit/database/test_trino_statement.py) and [`test_trino_connector.py`](packages/sources/tests/unit/database/test_trino_connector.py).
>
> **Not yet wired:** source-create validation and the SSRF guard, the serve-side executor and composite routing, infra, and UI — so the sub-type is **not usable end to end** on this branch yet.
>
> `authMode` is `BASIC` or `JWT`. OAuth2 client-credentials and mTLS are kept out of scope currently.
>
> Trino's own fine-grained access control **does not apply** to queries from this service, which connects under one service identity. Scope the credential to exactly what the source should expose.

## Architecture

The system follows a **Scan → Model → Serve** workflow:

- **Scan** — Connect data sources, discover schemas, enrich metadata, ingest unstructured documents
- **Model** — Induce and manage ontologies, define metrics, build a unified semantic graph
- **Serve** — Query via SPARQL federation (VKG), traverse the knowledge graph, serve context to AI agents via MCP

Access is governed by namespace isolation and role-based access control: namespace-scoped roles (owner, maintainer, data-steward, data-analyst) plus **platform-level roles** (`platform-admin`, `platform-viewer`) that apply across all namespaces. See the [control-plane docs](packages/control-plane/README.md) for the grants and authorization model.

## Quick Start

### Prerequisites

- Python 3.12, Node.js 22.23.4+, Docker
- [pnpm](https://pnpm.io/) (Node package manager — installed via `mise` or `npm install -g pnpm`)
- Java 17–24 (21 recommended) and Gradle (for Smithy codegen), plus Maven (for the
  Databricks connector `make deploy-dev` builds). JDK 25+ is not supported yet: the Athena
  Federation SDK the connector bundles fails its tests on it, and `make preflight` rejects it.
  **`brew install maven` also installs the newest OpenJDK and runs Maven on it** — after
  installing, check `mvn -v` and, if it shows Java 25+, point Maven at JDK 21:
  `export JAVA_HOME=$(/usr/libexec/java_home -v 21)` (`brew install --cask corretto@21` if
  you don't have it). `mise install` provides a supported JDK and Maven from `.mise.toml`.
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Setup

> **Use a release, not the tip of `main`.** We recommend starting from a tagged
> release — [github.com/aws/context-ontology-accelerator/releases](https://github.com/aws/context-ontology-accelerator/releases).
> `main` tracks ongoing development.

```bash
# Clone at the latest release tag (replace <tag> with a release from the link above)
git clone --branch <tag> https://github.com/aws/context-ontology-accelerator.git
cd context-ontology-accelerator

make setup      # install Python + CDK TypeScript dependencies
make format     # auto-format code
make lint       # check linting
make test       # run unit tests
```

**Full developer guide:** [external-docs/content/getting-started.md](external-docs/content/getting-started.md)

## Repository Structure

```
semantic-context/
├── models/              # Smithy API models (source of truth for API contracts)
├── smithy-generated/    # Auto-generated from Smithy (OpenAPI, Python interfaces, TS client)
├── infra/               # AWS CDK (TypeScript) — foundation + per-service stacks
├── packages/
│   ├── control-plane/           # Control Plane APIs
│   ├── data-layer/              # Data Layer APIs (query, retrieval, traversal)
│   ├── sources/                 # Unified data source ingestion (database + documents)
│   ├── ontology-engine/         # Ontology induction, reasoning (HermiT/ELK)
│   ├── metric-service/          # Metric authoring and resolution
│   ├── vkg/                     # Virtual Knowledge Graph (Ontop)
│   ├── mcp-server/              # MCP tools for AI agents
│   ├── context-manager/         # Serve layer Context Manager (query orchestration, upstream clients)
│   └── web-app/                 # React + Cloudscape frontend
├── libs/common/         # Shared Python config, logging, exceptions
├── libs/ts-shared/      # Shared TypeScript interfaces and constants
├── connectors/          # Athena federation connectors: toolkit, reference connector, Databricks
├── coa-skills/          # Agent skills for driving a deployed COA (control-plane)
├── scripts/             # CI-agnostic build/test/deploy scripts
└── external-docs/       # Published documentation (getting-started, deployment guides)
```

## Tech Stack

| Layer                  | Technology                                               |
| ---------------------- | -------------------------------------------------------- |
| Languages              | Python 3.12, TypeScript                                  |
| API Contracts          | Smithy → OpenAPI + Python interfaces + TypeScript client |
| IaC                    | AWS CDK (TypeScript)                                     |
| Frontend               | React + Cloudscape Design System                         |
| Package Management     | uv (Python), pnpm (TypeScript)                           |
| Monorepo Orchestration | Nx                                                       |
| Testing                | pytest (unit + integ)                                    |
| Linting                | ruff, mypy (strict)                                      |
| Codegen                | Smithy CLI (Java 17 + Gradle)                            |

## Contributing

This repository is published as a read-only mirror. We are not accepting pull requests at this time. You are welcome to report bugs and share feedback through GitHub Issues. See [CONTRIBUTING.md](CONTRIBUTING.md) for details including direction for Amazon employees.

## External Dependencies

This package depends on and may incorporate or retrieve a number of third-party
software packages (such as open source packages) at install-time or build-time
or run-time ("External Dependencies"). The External Dependencies are subject to
license terms that you must accept in order to use this package. If you do not
accept all of the applicable license terms, you should not use this package. We
recommend that you consult your company’s open source approval policy before
proceeding.

Provided below is a list of External Dependencies and the applicable license
identification as indicated by the documentation associated with the External
Dependencies as of Amazon's most recent review.

THIS INFORMATION IS PROVIDED FOR CONVENIENCE ONLY. AMAZON DOES NOT PROMISE THAT
THE LIST OR THE APPLICABLE TERMS AND CONDITIONS ARE COMPLETE, ACCURATE, OR
UP-TO-DATE, AND AMAZON WILL HAVE NO LIABILITY FOR ANY INACCURACIES. YOU SHOULD
CONSULT THE DOWNLOAD SITES FOR THE EXTERNAL DEPENDENCIES FOR THE MOST COMPLETE
AND UP-TO-DATE LICENSING INFORMATION.

YOUR USE OF THE EXTERNAL DEPENDENCIES IS AT YOUR SOLE RISK. IN NO EVENT WILL
AMAZON BE LIABLE FOR ANY DAMAGES, INCLUDING WITHOUT LIMITATION ANY DIRECT,
INDIRECT, CONSEQUENTIAL, SPECIAL, INCIDENTAL, OR PUNITIVE DAMAGES (INCLUDING
FOR ANY LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, OR
COMPUTER FAILURE OR MALFUNCTION) ARISING FROM OR RELATING TO THE EXTERNAL
DEPENDENCIES, HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, EVEN
IF AMAZON HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. THESE LIMITATIONS
AND DISCLAIMERS APPLY EXCEPT TO THE EXTENT PROHIBITED BY APPLICABLE LAW.

| External Dependency | License | Source |
| ------------------- | ------- | ------ |
| owlready2 | LGPL-3.0 | https://pypi.org/project/owlready2/ |

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
