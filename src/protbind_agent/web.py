"""Dependency-light, loopback-only research dashboard and local pose viewer."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .artifacts import ArtifactStore
from .control import StageControlStore
from .dossier import build_run_dossier, dossier_html
from .manifest import ManifestStore, RunManifest
from .models import ArtifactRef
from .pose_view import build_pose_scene_summary
from .workflow import ProtBindWorkflow


def _page(title: str, body: str) -> str:
    navigation = " ".join(
        f"<a href='{path}'>{label}</a>"
        for path, label in (
            ("/cases", "Cases"),
            ("/dossiers", "Run dossiers"),
            ("/funnel", "Screening funnel"),
            ("/poses", "3D poses"),
            ("/evidence", "Evidence"),
            ("/rag", "RAG"),
            ("/performance", "Radeon performance"),
        )
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; object-src 'none'\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;"
        "color:#172033}nav{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:2rem}"
        "a{color:#1459b8}code,pre{background:#f3f4f6;padding:.2rem .4rem}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #d8dee9;"
        "padding:.55rem;text-align:left;vertical-align:top}.muted{color:#596579}"
        ".warning{border-left:4px solid #c77800;padding:.75rem;background:#fff7e6}"
        ".viewer{width:100%;height:680px;border:1px solid #ccd4df;border-radius:.5rem}"
        ".metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));"
        "gap:.7rem;margin:1rem 0}.metric{background:#f6f8fb;padding:.75rem;border-radius:.4rem}"
        "button{padding:.55rem .8rem;margin:.4rem .4rem .4rem 0}</style></head>"
        f"<body><nav>{navigation}</nav><h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def _manifest_paths(root: Path) -> list[Path]:
    return sorted((root / "runs").glob("*/manifest.json"))


def create_app(workspace: Path) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for 'protbind serve'") from exc

    root = workspace.resolve()
    artifacts = ArtifactStore(root)
    manifests = ManifestStore(root)
    control = StageControlStore(ProtBindWorkflow(root))
    app = FastAPI(title="ProtBind local research agent", docs_url=None, redoc_url=None)

    def load_manifest(run_id: str) -> RunManifest:
        try:
            return manifests.load(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="run manifest is invalid") from exc

    def pose_summary(run_id: str) -> dict[str, Any]:
        manifest = load_manifest(run_id)
        try:
            return build_pose_scene_summary(manifest, artifacts)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"pose scene is unavailable: {type(exc).__name__}",
            ) from exc

    def selected_scene(run_id: str, candidate_id: str) -> dict[str, Any]:
        summary = pose_summary(run_id)
        for scene in summary["candidates"]:
            if scene["candidate_id"] == candidate_id:
                return scene
        raise HTTPException(status_code=404, detail="pose candidate not found")

    def run_rows() -> str:
        rows = []
        for path in _manifest_paths(root):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                run_id = str(value["run_id"])
                encoded = quote(run_id, safe="")
                rows.append(
                    "<tr>"
                    f"<td><code>{html.escape(run_id)}</code></td>"
                    f"<td>{html.escape(str(value['case_id']))}</td>"
                    f"<td>{html.escape(str(value['state']))}</td>"
                    f"<td>{html.escape(str(value['last_completed_stage']))}</td>"
                    f"<td><a href='/runs/{encoded}/dossier'>dossier</a> · "
                    f"<a href='/runs/{encoded}/poses'>poses</a></td>"
                    "</tr>"
                )
            except (KeyError, OSError, json.JSONDecodeError):
                continue
        return "".join(rows) or "<tr><td colspan='5'>No runs yet.</td></tr>"

    @app.get("/", response_class=HTMLResponse)
    @app.get("/cases", response_class=HTMLResponse)
    def cases() -> str:
        return _page(
            "Research cases",
            "<table hx-get='/fragments/runs' hx-trigger='every 5s' hx-swap='innerHTML'>"
            "<thead><tr><th>Run</th><th>Case</th><th>State</th><th>Last stage</th>"
            f"<th>Inspect</th></tr></thead><tbody>{run_rows()}</tbody></table>",
        )

    @app.get("/fragments/runs", response_class=HTMLResponse)
    def run_fragment() -> str:
        return run_rows()

    @app.get("/dossiers", response_class=HTMLResponse)
    def dossiers() -> str:
        return _page(
            "Run completion dossiers",
            "<p>Each dossier distinguishes core computation from stage-gate acceptance and "
            "binds every displayed result to content-addressed artifacts.</p>"
            "<table><thead><tr><th>Run</th><th>Case</th><th>State</th>"
            f"<th>Last stage</th><th>Inspect</th></tr></thead><tbody>{run_rows()}</tbody></table>",
        )

    @app.get("/runs/{run_id}/dossier", response_class=HTMLResponse)
    def run_dossier(run_id: str) -> str:
        manifest = load_manifest(run_id)
        poses = build_pose_scene_summary(manifest, artifacts)
        dossier = build_run_dossier(
            manifest,
            artifacts,
            control_history=control.read(run_id),
            pose_summary=poses,
        )
        return dossier_html(dossier)

    @app.get("/funnel", response_class=HTMLResponse)
    def funnel() -> str:
        return _page(
            "Screening funnel",
            "<p>100,000 → TriPharm 512 → scaffold diversity 128 → quick Vina 16 → "
            "evidence-grade Vina 16 → optional complex-prediction cross-check 8 → "
            "evidence report 5.</p><p>Run artifacts provide the measured counts; this page "
            "never substitutes planned counts for results. Complex prediction is not required "
            "for docking or validation.</p>",
        )

    @app.get("/poses", response_class=HTMLResponse)
    def poses() -> str:
        asset = root / "static" / "3Dmol-min.js"
        status = (
            "Pinned local 3Dmol.js asset is installed."
            if asset.is_file()
            else (
                "3Dmol.js is not installed locally; remote CDN loading is disabled. "
                "Pose metadata remains available, but interactive rendering is unavailable."
            )
        )
        rows: list[str] = []
        for path in _manifest_paths(root):
            try:
                manifest = manifests.load(path.parent.name)
                summary = build_pose_scene_summary(
                    manifest,
                    artifacts,
                    include_geometry=False,
                )
                for scene in summary["candidates"]:
                    run_id = quote(manifest.run_id, safe="")
                    candidate = quote(scene["candidate_id"], safe="")
                    rows.append(
                        "<tr>"
                        f"<td><code>{html.escape(manifest.run_id)}</code></td>"
                        f"<td>{html.escape(scene['candidate_id'])}</td>"
                        f"<td>{html.escape(scene['molecule_id'])}</td>"
                        f"<td>{html.escape(str(scene['vina_score']))}</td>"
                        f"<td><a href='/runs/{run_id}/poses/{candidate}'>open locally</a></td>"
                        "</tr>"
                    )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                continue
        table_rows = "".join(rows) or "<tr><td colspan='5'>No docked poses yet.</td></tr>"
        return _page(
            "3D poses",
            f"<p>{html.escape(status)}</p>"
            "<p class='warning'>Visual inspection is QA only. It does not replace "
            "PoseBusters, ProLIF, symmetry-aware RMSD, or physical validation.</p>"
            "<table><thead><tr><th>Run</th><th>Candidate</th><th>Molecule</th>"
            f"<th>Vina tool score</th><th>Viewer</th></tr></thead><tbody>{table_rows}</tbody>"
            "</table>",
        )

    @app.get("/runs/{run_id}/poses", response_class=HTMLResponse)
    def run_poses(run_id: str) -> str:
        summary = pose_summary(run_id)
        rows = []
        for scene in summary["candidates"]:
            candidate = quote(scene["candidate_id"], safe="")
            rows.append(
                "<tr>"
                f"<td>{html.escape(scene['candidate_id'])}</td>"
                f"<td>{html.escape(scene['molecule_id'])}</td>"
                f"<td>{html.escape(str(scene['vina_score']))}</td>"
                "<td>"
                f"{html.escape(str(scene['validation'].get('posebusters_valid')))}</td>"
                f"<td><a href='/runs/{quote(run_id, safe='')}/poses/{candidate}'>view</a></td>"
                "</tr>"
            )
        return _page(
            f"Docked poses — {run_id}",
            "<table><thead><tr><th>Candidate</th><th>Molecule</th>"
            "<th>Vina tool score</th><th>PB-valid</th><th>Local view</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>",
        )

    @app.get("/runs/{run_id}/poses/{candidate_id}", response_class=HTMLResponse)
    def pose_page(run_id: str, candidate_id: str) -> str:
        scene = selected_scene(run_id, candidate_id)
        geometry = scene["geometry"]
        validation = scene["validation"]
        metrics = (
            f"<div class='metric'><strong>Vina tool score</strong><br>"
            f"{html.escape(str(scene['vina_score']))}</div>"
            f"<div class='metric'><strong>PoseBusters valid</strong><br>"
            f"{html.escape(str(validation.get('posebusters_valid')))}</div>"
            f"<div class='metric'><strong>Evidence grade</strong><br>"
            f"{html.escape(str(validation.get('evidence_grade')))}</div>"
            f"<div class='metric'><strong>Minimum heavy-atom distance</strong><br>"
            f"{html.escape(str(geometry.get('minimum_heavy_atom_distance_angstrom')))} Å</div>"
            f"<div class='metric'><strong>Sub-2 Å pairs</strong><br>"
            f"{html.escape(str(geometry.get('sub_2_angstrom_pair_count')))}</div>"
            f"<div class='metric'><strong>Inside docking box</strong><br>"
            f"{html.escape(str(geometry.get('all_ligand_heavy_atoms_inside_declared_box')))}</div>"
        )
        scene_json = json.dumps(
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "receptor_format": scene["receptor_format"],
                "pose_format": scene["pose_format"],
                "box_center": scene["box_center"],
                "box_size": scene["box_size"],
            },
            ensure_ascii=False,
        ).replace("</", "<\\/")
        body = (
            "<p class='warning'>This image is a local visual-QA view, not evidence that the "
            "ligand binds experimentally.</p>"
            f"<div class='metrics'>{metrics}</div>"
            "<button id='reset-view'>Reset view</button>"
            "<button id='download-png'>Download local PNG</button>"
            "<div id='viewer' class='viewer'></div>"
            "<script src='/static/3Dmol-min.js'></script>"
            f"<script>const scene={scene_json};"
            "async function boot(){"
            "if(typeof $3Dmol==='undefined'){document.getElementById('viewer').textContent="
            "'Local 3Dmol.js asset is missing.';return;}"
            "const base='/api/runs/'+encodeURIComponent(scene.run_id)+'/poses/'+"
            "encodeURIComponent(scene.candidate_id);"
            "const [receptor,ligand]=await Promise.all([fetch(base+'/receptor').then(r=>r.text()),"
            "fetch(base+'/ligand').then(r=>r.text())]);"
            "const viewer=$3Dmol.createViewer('viewer',{backgroundColor:'white'});"
            "viewer.addModel(receptor,scene.receptor_format);"
            "viewer.setStyle({model:0},{cartoon:{color:'spectrum',opacity:0.82}});"
            "viewer.addModel(ligand,scene.pose_format);"
            "viewer.setStyle({model:1},{stick:{colorscheme:'greenCarbon',radius:0.22},"
            "sphere:{colorscheme:'greenCarbon',scale:0.28}});"
            "viewer.setStyle({model:0,within:{distance:5,sel:{model:1}}},"
            "{stick:{colorscheme:'cyanCarbon',radius:0.16}});"
            "viewer.addBox({center:{x:scene.box_center[0],y:scene.box_center[1],"
            "z:scene.box_center[2]},dimensions:{w:scene.box_size[0],h:scene.box_size[1],"
            "d:scene.box_size[2]},color:'orange',wireframe:true});"
            "viewer.zoomTo({model:1});viewer.render();"
            "document.getElementById('reset-view').onclick=()=>{viewer.zoomTo({model:1});"
            "viewer.render();};"
            "document.getElementById('download-png').onclick=()=>{const a=document."
            "createElement('a');a.href=viewer.pngURI();a.download='protbind-'+"
            "scene.candidate_id+'.png';a.click();};"
            "}boot().catch(e=>{document.getElementById('viewer').textContent="
            "'Viewer failed: '+e.message;});</script>"
        )
        return _page(f"Pose — {candidate_id}", body)

    @app.get("/static/3Dmol-min.js")
    def three_dmol() -> Response:
        asset = root / "static" / "3Dmol-min.js"
        if not asset.is_file():
            raise HTTPException(status_code=404, detail="local 3Dmol.js asset not installed")
        return Response(
            asset.read_bytes(),
            media_type="text/javascript",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/runs/{run_id}/poses/{candidate_id}", response_class=JSONResponse)
    def pose_metadata(run_id: str, candidate_id: str) -> dict[str, Any]:
        return selected_scene(run_id, candidate_id)

    @app.get("/api/runs/{run_id}/poses/{candidate_id}/receptor")
    def pose_receptor(run_id: str, candidate_id: str) -> Response:
        scene = selected_scene(run_id, candidate_id)
        reference = ArtifactRef.from_dict(scene["receptor"])
        return Response(
            artifacts.read_bytes(reference),
            media_type=reference.media_type,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/api/runs/{run_id}/poses/{candidate_id}/ligand")
    def pose_ligand(run_id: str, candidate_id: str) -> Response:
        scene = selected_scene(run_id, candidate_id)
        reference = ArtifactRef.from_dict(scene["pose"])
        return Response(
            artifacts.read_bytes(reference),
            media_type=reference.media_type,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/evidence", response_class=HTMLResponse)
    def evidence() -> str:
        return _page(
            "Evidence",
            "<p>Grades: REDOCKING_RECOVERED, METHOD_CONSENSUS, HYPOTHESIS_ONLY, and "
            "REJECTED. Redocking recovery is a method-control result, not evidence that a "
            "compound binds experimentally. Resolve every citation through its SHA-256 "
            "artifact ID.</p>",
        )

    @app.get("/rag", response_class=HTMLResponse)
    def rag() -> str:
        return _page(
            "Private RAG",
            "<p>seekdb/PowerMem is capability-gated. No document or sequence is uploaded "
            "without explicit approval.</p>",
        )

    @app.get("/performance", response_class=HTMLResponse)
    def performance() -> str:
        return _page(
            "Radeon performance",
            "<p>Scientific kernels, OpenFold3, OpenMM, and HipFire metrics are reported "
            "separately. Architecture spoofing via HSA_OVERRIDE_GFX_VERSION is forbidden.</p>",
        )

    return app
