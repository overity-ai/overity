"""
Overity.ai report encoder/decoder
=================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from overity.model.report import (
    MethodExecutionStage,
    MethodExecutionStatus,
    MethodReportLogItem,
    MethodReport,
)
from overity.model.general_info.method import MethodKind, MethodAuthor, MethodInfo
from overity.model.traceability import (
    ArtifactKey,
    ArtifactGraph,
    ArtifactLink,
    ArtifactLinkKind,
    ArtifactKind,
)
from overity.model.report.metrics import Metric, from_data as metrics_from_data
from datetime import datetime as dt


# --------------------------- Encoder


def _encode_method_author(author: MethodAuthor) -> dict[str, str]:
    x = {
        "name": author.name,
        "email": author.email,
    }

    if author.contribution is not None:
        x["contribution"] = author.contribution

    return x


def _encode_method_info(method: MethodInfo) -> dict[str, Any]:
    x = {
        "slug": method.slug,
        "kind": method.kind.value,
        "display_name": method.display_name,
        "authors": [_encode_method_author(x) for x in method.authors],
        "metadata": method.metadata,
    }

    if method.description:
        x["description"] = method.description

    if method.path:
        x["path"] = str(method.path)

    return x


def _encode_artifact_key(x: ArtifactKey) -> dict[str, str]:
    return {
        "kind": x.kind.value,
        "id": x.id,
    }


def _encode_traceability_graph(gr: ArtifactGraph) -> dict[str, Any]:
    def do_item(x):
        return {
            "a": _encode_artifact_key(x.a),
            "b": _encode_artifact_key(x.b),
            "kind": x.kind.value,
        }

    return {
        "links": [do_item(link) for link in gr.links],
        "metadata": [
            {
                "kind": key.kind.value,
                "id": key.id,
                "data": data,
            }
            for key, data in gr.metadata.items()
        ],
    }


def _encode_logs(x: list[MethodReportLogItem]) -> list[dict[str, Any]]:
    def do_item(x):
        return {
            "dt": x.timestamp.isoformat(),
            "severity": x.severity,
            "source": x.source,
            "message": x.message,
        }

    return [do_item(item) for item in x]


def _encode_metrics(x: dict[str, Metric]) -> dict[str, Any]:
    return {k: v.data() for k, v in x.items()}


def to_file(report: MethodReport, path: Path):
    output_obj = {
        "uuid": report.uuid,
        "program": report.program,
        "date_started": report.date_started.isoformat(),
        "date_ended": report.date_ended.isoformat(),
        "stage": report.stage.value,
        "status": report.status.value,
        "environment": report.environment,
        "context": report.context,
        "method_info": _encode_method_info(report.method_info),
        "traceability_graph": _encode_traceability_graph(report.traceability_graph),
        "logs": _encode_logs(report.logs),
        "metrics": _encode_metrics(report.metrics or {}),
        # outputs TODO #
    }

    with open(path, "w") as fhandle:
        json.dump(output_obj, fhandle)


# --------------------------- Decoder


def _parse_method_author(data: dict[str, str]) -> MethodAuthor:
    return MethodAuthor(
        name=data["name"],
        email=data["email"],
        contribution=data.get("contribution", None),
    )


def _parse_method_info(data: dict[str, Any]) -> MethodInfo:
    return MethodInfo(
        slug=data["slug"],
        kind=MethodKind(data["kind"]),
        display_name=data["display_name"],
        authors=[_parse_method_author(x) for x in data["authors"]],
        metadata=data["metadata"],
        description=data.get("description", None),
        path=Path(data["path"]) if data.get("path") else None,
    )


def _parse_artifact_key(data: dict[str, Any]) -> ArtifactKey:
    return ArtifactKey(kind=ArtifactKind(data["kind"]), id=data["id"])


def _parse_traceability_graph(data: dict[str, Any]) -> ArtifactGraph:
    def process_item(x):
        return ArtifactLink(
            a=_parse_artifact_key(x["a"]),
            b=_parse_artifact_key(x["b"]),
            kind=ArtifactLinkKind(x["kind"]),
        )

    def process_metadata(metadata_list):
        parsed_dict = {
            _parse_artifact_key(item): item["data"] for item in metadata_list
        }
        return parsed_dict

    return ArtifactGraph(
        links={process_item(link) for link in data["links"]},
        metadata=process_metadata(data["metadata"]),
    )


def _parse_logs(data: list[dict[str, Any]]) -> list[MethodReportLogItem]:
    def process_item(x):
        return MethodReportLogItem(
            timestamp=dt.fromisoformat(x["dt"]),
            severity=x["severity"],
            source=x["source"],
            message=x["message"],
        )

    return [process_item(item) for item in data]


def _parse_metrics(data: dict[str, dict[str, Any]]) -> dict[str, Metric]:
    return {k: metrics_from_data(v) for k, v in data.items()}


def from_file(path: Path):
    path = Path(path)

    with open(path, "r") as fhandle:
        data = json.load(fhandle)

    # TODO: Schema validation
    out_report = MethodReport(
        uuid=data["uuid"],
        program=data["program"],
        date_started=dt.fromisoformat(data["date_started"]),
        date_ended=dt.fromisoformat(data["date_ended"]),
        stage=MethodExecutionStage(data["stage"]),
        status=MethodExecutionStatus(data["status"]),
        environment=data["environment"],
        context=data["context"],
        method_info=_parse_method_info(data["method_info"]),
        traceability_graph=_parse_traceability_graph(data["traceability_graph"]),
        logs=_parse_logs(data["logs"]),
        metrics=_parse_metrics(data["metrics"]),
        outputs=None,  # TODO: Parse outputs
    )

    return out_report
