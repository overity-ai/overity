"""
Report HTML view for training/optimization reports
==================================================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

# TODO: Embed external resources

from __future__ import annotations

import datetime as dt

from pathlib import Path
from textwrap import dedent
from io import StringIO

from jinja2 import Template

from overity.model.traceability import ArtifactGraph, ArtifactKind
from overity.model.report import (
    MethodReport,
    MethodExecutionStatus,
    MethodExecutionStage,
)
from overity.model.report.metrics import (
    Metric,
    SimpleValue,
    LinScaleValue,
    LinRangeValue,
    PercentageValue,
)


TEMPLATE_TXT = dedent(
    """\
    <!doctype html>
    <html>
        <head>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1"/>

            <title>Report</title>


            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-SgOJa3DmI69IUzQ2PVdRZhwQ+dy64/BUtbMJw1MZ8t5HZApcHrRKUc4W0kG879m7" crossorigin="anonymous">

            <style>
                .report-header {
                    border-bottom: 3px solid #dee2e6;
                    padding: 20px;
                }
                .section-title {
                    border-left: 4px solid #0d6efd;
                    padding-left: 15px;
                    margin: 20px 0;
                }
                .report-body {
                    padding: 20px;
                }

                table {
                    margin-top: 20px;
                }

                .mermaid {
                    text-align: center;
                }

                .log-view {
                    font-size: 0.9rem;
                }
                .log-view td {
                    padding: 0.2rem;
                }
                .log-view .severity-debug {
                    background-color: #f0f0f0;
                    color: #6c757d;
                }
                .log-view .severity-info {
                    background-color: #e5f2ff;
                    color: #0d6efd;
                }
                .log-view .severity-warning {
                    background-color: #fff8e2;
                    color: #ffc107;
                }
                .log-view .severity-error {
                    background-color: #ffe5e5;
                    color: #dc3545;
                }
                .log-view .severity-critical {
                    background-color: #ffd7d7;
                    color: #dc3545;
                }
            </style>
        </head>

        <body>
            <main role="main" class="container">
                <!-- Report header -->
                <div class="report-header">
                    <h1 class="mb-3">Overity.ai: Training optimization report view</h1>
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Report ID:</strong> {{report_id}}</p>
                        </div>

                        <div class="col-md-6 text-end">
                            <p><strong>Program:</strong> {{program_slug}}</p>
                        </div>
                    </div>
                </div>

                <!-- Report body -->
                <div class="report-body">

                    <!-- -------------------------- -->

                    <div class="section-title">
                        <h2>1. Identification</h2>
                    </div>

                    <div class="row">
                        <h3>1.1 Execution information
                    </div>

                    <div class="row">
                        <div class="col-md-12">
                            <ul>
                                <li><strong>Path:</strong> {{ report_path }}</li>
                                <li><strong>Started:</strong> {{ date_started }}</li>
                                <li><strong>Ended:</strong> {{ date_ended }}</li>
                                <li><strong>Duration:</strong> {{ duration }}</li>
                                <li><strong>Stage:</strong> {{ execution_stage }}</li>
                                <li><strong>Status:</strong> {{ execution_status }}</li>
                            </ul>
                        </div>
                    </div>

                    <div class="row">
                        <h3>1.2 Method information</h3>
                    </div>

                    <div class="row">
                        <div class="col-md-12">
                            <ul>
                                <li><strong>Method slug:</strong> {{ method_slug }}</li>
                                <li><strong>Method name:</strong> {{ method_name }}</li>
                            </ul>
                        </div>
                    </div>


                    <!-- -------------------------- -->
                    
                    <div class="section-title">
                        <h2>2. Environment</h2>
                    </div>

                    <h3>2.1 Installed packages</h3>

                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Version</th>
                            </tr>
                        </thead>

                        <tbody>
                            {% for item in installed_packages %}
                                <tr>
                                    <td>{{ item.name }}</td>
                                    <td>{{ item.version }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>

                    <h3>2.2 Misc. info</h3>
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Key</th>
                                <th>Value</th>
                            </tr>
                        </thead>

                        <tbody>
                            {% for item in environment %}
                                <tr>
                                    <td>{{ item.key }}</td>
                                    <td>{{ item.value }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>

                    <!-- -------------------------- -->


                    <div class="section-title">
                        <h2>3. Context</h2>
                    </div>

                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Key</th>
                                <th>Value</th>
                            </tr>
                        </thead>

                        <tbody>
                            {% for item in context %}
                                <tr>
                                    <td>{{ item.key }}</td>
                                    <td>{{ item.value }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>

                    <!-- -------------------------- -->

                    <div class="section-title">
                        <h2>4. Metrics</h2>
                    </div>

                    <div class="row">
                        {% for metric in metrics %}
                        <div class="col-xl-3 col-md-6 mb-4">
                            <div class="card border-left-primary shadow h-100 py-2">
                                <div class="card-body">
                                    <div class="row no-gutters align-items-center">
                                        <div class="col mr-2">
                                            <div class="text-xs font-weight-bold text-primary text-uppercase mb-1">{{metric.name}}</div>
                                            <div class="h5 mb-0 font-weigh-bold text-gray-800">{{metric.value}}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>



                    <!-- -------------------------- -->

                    <div class="section-title">
                        <h2>5. Traceability</h2>
                    </div>

                    <div class="row">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-body">
                                    <pre class="mermaid">
    {{ traceability_graph }}
                                    </pre>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- -------------------------- -->

                    <div class="section-title">
                        <h2>6. Logs</h2>
                    </div>

                    <div class="table-responsive">
                        <table class="table table-sm log-view">
                            <thead>
                                <th style="width: 5%">#</th>
                                <th style="width: 20%">Timestamp</th>
                                <th style="width: 10%">Severity</th>
                                <th style="width: 10%">Source</th>
                                <th>Message</th>
                            </thead>

                        <tbody>
                            {% for item in logs %}
                                <tr>
                                    <td>#{{ loop.index }}</td>
                                    <td>{{ item.timestamp }}</td>
                                    <td>{{ item.severity }}</td>
                                    <td>{{ item.source }}</td>
                                    <td><pre>{{ item.message }}</pre></td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>

                    <!-- -------------------------- -->

                    <div class="report-footer mt-5">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Generated:</strong> {{generation_dt}}</p>
                            </div>

                            <div class="col-md-6 text-end">
                                <p>Text TODO</p>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/js/bootstrap.bundle.min.js" integrity="sha384-k6d4wzSIapyDyv1kpU366/PK5hCdSbCRGRCMv+eplOQJWyd1fbcAu9OCUj5zNLiq" crossorigin="anonymous"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

                mermaid.initialize({
                    startOnLoad: true,
                    theme: "neutral",
                });
            </script>
        </body>
    </html>
"""
)


def _generate_traceability_graph(graph: ArtifactGraph):
    def name_clean(x):
        return x.lower().replace("-", "_")

    output = StringIO()

    print("graph TD", file=output)
    for nd in graph.nodes:
        stl = ""
        enl = ""

        # / shape /
        if nd.kind in {
            ArtifactKind.AnalysisMethod,
            ArtifactKind.DeploymentMethod,
            ArtifactKind.MeasurementQualificationMethod,
            ArtifactKind.TrainingOptimizationMethod,
        }:
            stl = "[/"
            enl = "/]"
        # Database shape
        elif nd.kind == ArtifactKind.Dataset:
            stl = "[("
            enl = ")]"
        # ( shape )
        elif nd.kind == ArtifactKind.Model:
            stl = "(["
            enl = "])"
        # [[ shape ]]
        elif nd.kind in {
            ArtifactKind.AnalysisReport,
            ArtifactKind.ExecutionReport,
            ArtifactKind.ExperimentRun,
            ArtifactKind.OptimizationReport,
        }:
            stl = "[["
            enl = "]]"
        # [ shape ] (standard box)
        else:
            stl = "["
            enl = "]"

        label_str = f"({nd.kind.value})<br>id: {nd.id}"
        if nd in graph.metadata:
            for k, v in graph.metadata[nd].items():
                label_str += f"<br>{k}: {v}"

        print(
            f'    {name_clean(nd.kind.value)}_{name_clean(nd.id)}{stl}"{label_str}"{enl}',
            file=output,
        )

    for lk in graph.links:
        a_id = f"{name_clean(lk.a.kind.value)}_{name_clean(lk.a.id)}"
        b_id = f"{name_clean(lk.b.kind.value)}_{name_clean(lk.b.id)}"

        print(f"    {a_id} -->|{lk.kind.value}| {b_id}", file=output)

    return output.getvalue()


def _format_duration(x: dt.timedelta) -> str:
    return f"{x.days} days {x.seconds} seconds"


def _execution_status_str(x: MethodExecutionStatus) -> str:
    results = {
        MethodExecutionStatus.ExecutionSuccess: "Success",
        MethodExecutionStatus.ExecutionFailureException: "Failed with errors",
        MethodExecutionStatus.ExecutionFailureConstraints: "Failed to meet constraints",
    }

    return results[x]


def _execution_stage_str(x: MethodExecutionStage) -> str:
    results = {
        MethodExecutionStage.Preview: "Preview",
        MethodExecutionStage.Operation: "Operation",
    }

    return results[x]


def _process_metric(x: Metric):
    if isinstance(x, SimpleValue):
        return f"{x.value:.2f}"
    elif isinstance(x, LinScaleValue):
        return f"{x.value:.2f} ({x.low:.2f} / {x.high:.2f})"
    elif isinstance(x, LinRangeValue):
        return f"{x.value} ({x.low} / {x.high})"
    elif isinstance(x, PercentageValue):
        return f"{x.value*100:.2f} %"


def render(report_data: MethodReport, report_path: Path | None = None):
    template = Template(TEMPLATE_TXT)

    # Filter environment information
    environment_no_pkgs = {
        k: v for k, v in report_data.environment.items() if k != "installed_packages"
    }
    installed_packages = [
        {"name": v[0], "version": v[1] if len(v) > 1 else "N.A."}
        for v in map(
            lambda x: x.split("=="), report_data.environment["installed_packages"]
        )
    ]

    template_in_data = {
        "report_id": report_data.uuid,
        "report_path": report_path,
        "program_slug": report_data.program,
        "method_slug": report_data.method_info.slug,
        "method_name": report_data.method_info.display_name,
        "generation_dt": dt.datetime.now().isoformat(),
        "date_started": report_data.date_started,
        "date_ended": report_data.date_ended,
        "execution_stage": _execution_stage_str(report_data.stage),
        "execution_status": _execution_status_str(report_data.status),
        "duration": _format_duration(report_data.date_ended - report_data.date_started),
        "installed_packages": installed_packages,
        "environment": [{"key": k, "value": v} for k, v in environment_no_pkgs.items()],
        "context": [{"key": k, "value": v} for k, v in report_data.context.items()],
        "traceability_graph": _generate_traceability_graph(
            report_data.traceability_graph
        ),
        "logs": [
            {
                "timestamp": it.timestamp,
                "severity": it.severity,
                "source": it.source,
                "message": it.message,
            }
            for it in report_data.logs
        ],
        "metrics": [
            {"name": k, "value": _process_metric(v)}
            for k, v in report_data.metrics.items()
        ],
    }

    return template.render(**template_in_data)
