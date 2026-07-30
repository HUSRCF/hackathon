from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from radeon_agent.opencode_deepseek import (
    DEEPSEEK_HOST,
    _validate_passthrough,
    build_inline_config,
    prepare_environment,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_opencode_configuration_is_local_and_default_deny() -> None:
    config = json.loads((REPOSITORY_ROOT / "opencode.json").read_text(encoding="utf-8"))

    assert config["share"] == "disabled"
    assert config["autoupdate"] is False
    assert config["subagent_depth"] == 0
    assert config["enabled_providers"] == ["hipfire"]
    assert config["provider"]["hipfire"]["options"]["baseURL"].startswith(
        "http://127.0.0.1:"
    )
    assert set(config["provider"]["hipfire"]["models"]) == {"qwen3.5:9b"}
    assert config["permission"]["*"] == "deny"
    assert config["permission"]["protbind_case_dossier"] == "allow"
    assert config["permission"]["protbind_case_pose_view"] == "allow"
    assert config["permission"]["protbind_fetch_public_data"] == "ask"
    assert config["permission"]["protbind_case_advance"] == "ask"
    assert config["permission"]["protbind_case_create"] == "ask"
    assert config["permission"]["protbind_case_attach_support"] == "ask"
    for name in (
        "protbind_library_status",
        "protbind_library_list",
        "protbind_library_show",
        "protbind_library_plan_import",
        "protbind_library_apply_import",
        "protbind_library_verify_uniprot",
        "protbind_knowledge_document_inspect",
        "protbind_knowledge_import",
        "protbind_knowledge_search",
        "protbind_library_rag_sync",
        "protbind_library_rag_search",
    ):
        assert config["permission"][name] == "ask"
    assert config["permission"]["protbind_knowledge_model_status"] == "allow"
    assert config["permission"]["skill"]["protbind-library"] == "allow"


def test_opencode_shadowplan_plugin_is_redacted_and_event_complete() -> None:
    plugin = REPOSITORY_ROOT / ".opencode/plugins/protbind-shadowplan.ts"
    text = plugin.read_text(encoding="utf-8")

    for hook in (
        '"permission.ask"',
        '"permission.updated"',
        '"permission.replied"',
        '"tool.execute.before"',
        '"tool.execute.after"',
        '"session.status"',
        '"session.idle"',
    ):
        assert hook in text
    assert "protbind_case_advance" in text
    assert "continuation-token-use" in text
    assert "Bun.$" not in text

    script = """
import {
  buildOpenCodeShadowPlan,
  ProtBindShadowPlan
} from "./.opencode/plugins/protbind-shadowplan.ts";
const plan = await buildOpenCodeShadowPlan(
  "protbind_case_advance",
  {run_id: "run-1", continuation_token: "private-token"}
);
const toasts = [];
const logs = [];
const hooks = await ProtBindShadowPlan({
  client: {
    tui: {showToast: async (value) => {toasts.push(value)}},
    app: {log: async (value) => {logs.push(value)}},
  },
  directory: ".",
});
await hooks["permission.ask"]({
  id: "permission-1",
  type: "protbind_case_advance",
  sessionID: "session-1",
  callID: "call-1",
  metadata: {args: {continuation_token: "private-token"}},
}, {status: "ask"});
await hooks.event({
  event: {
    type: "permission.replied",
    properties: {
      sessionID: "session-1",
      permissionID: "permission-1",
      response: "once",
    },
  },
});
await hooks["tool.execute.before"]({
  tool: "protbind_case_advance",
  sessionID: "session-1",
  callID: "call-1",
}, {args: {}});
const output = {title: "advance", output: "ok", metadata: {}};
await hooks["tool.execute.after"]({
  tool: "protbind_case_advance",
  sessionID: "session-1",
  callID: "call-1",
  args: {},
}, output);
console.log(JSON.stringify({
  status: plan.status,
  digestLength: plan.arguments_sha256.length,
  leaked: JSON.stringify(plan).includes("private-token"),
  postflight: plan.safe_idle_tasks.includes("compile-one-stage-postflight-checklist"),
  pluginStatus: output.metadata.protbind_shadow_plan.status,
  pluginLeaked: JSON.stringify({toasts, logs, output}).includes("private-token"),
}));
"""
    completed = subprocess.run(
        ["bun", "-e", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "status": "WAITING_APPROVAL",
        "digestLength": 64,
        "leaked": False,
        "postflight": True,
        "pluginStatus": "EXECUTED",
        "pluginLeaked": False,
    }


def test_opencode_mcp_uses_aiaa_stdio_and_no_worker_placeholders() -> None:
    config = json.loads((REPOSITORY_ROOT / "opencode.json").read_text(encoding="utf-8"))
    mcp = config["mcp"]["protbind"]
    command = mcp["command"]

    assert mcp["type"] == "local"
    assert mcp["cwd"] == "."
    assert command[:4] == [
        "scripts/aiaa-protbind.sh",
        "-m",
        "protbind_agent",
        "mcp",
    ]
    assert "serve" in command
    assert "--worker-config" not in command
    assert command[-4:] == [
        "--library-config",
        ".protbind/library.json",
        "--knowledge-model",
        ".protbind/models/bge-m3",
    ]
    assert mcp["environment"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_project_skill_declares_only_portable_frontmatter() -> None:
    skill = (
        REPOSITORY_ROOT / ".agents/skills/protbind-research/SKILL.md"
    ).read_text(encoding="utf-8")
    frontmatter = skill.split("---", maxsplit=2)[1]
    keys = {
        line.split(":", maxsplit=1)[0].strip()
        for line in frontmatter.splitlines()
        if ":" in line
    }

    assert keys == {"name", "description"}
    assert "Call `protbind_case_status` before every attempted advance." in skill
    assert "Never auto-retry a failure." in skill


def test_library_skill_requires_fresh_consent_and_no_arbitrary_paths() -> None:
    skill = (
        REPOSITORY_ROOT / ".agents/skills/protbind-library/SKILL.md"
    ).read_text(encoding="utf-8")
    frontmatter = skill.split("---", maxsplit=2)[1]
    keys = {
        line.split(":", maxsplit=1)[0].strip()
        for line in frontmatter.splitlines()
        if ":" in line
    }

    assert keys == {"name", "description"}
    assert "fresh confirmation" in skill
    assert "cannot browse an arbitrary source path" in skill
    assert "Default to `mode=copy`" in skill


def test_deepseek_opencode_override_is_explicit_and_secret_free() -> None:
    config = build_inline_config()

    assert config["enabled_providers"] == ["deepseek-cloud"]
    assert config["model"] == "deepseek-cloud/deepseek-v4-flash"
    assert config["small_model"] == config["model"]
    provider = config["provider"]["deepseek-cloud"]
    assert provider["options"]["baseURL"] == "https://api.deepseek.com"
    assert provider["options"]["apiKey"] == "{env:DEEPSEEK_API_KEY}"
    assert (
        provider["models"]["deepseek-v4-flash"]["options"]["thinking"]["type"]
        == "disabled"
    )
    assert "secret-value" not in json.dumps(config)


def test_deepseek_opencode_launcher_requires_exact_domain_and_env_precedence(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=file-secret\n", encoding="utf-8")

    child = prepare_environment(
        env={"DEEPSEEK_API_KEY": "environment-secret"},
        env_file=env_file,
        approved_domains=[DEEPSEEK_HOST],
        model="deepseek-v4-flash",
    )

    assert child["DEEPSEEK_API_KEY"] == "environment-secret"
    assert "environment-secret" not in child["OPENCODE_CONFIG_CONTENT"]
    with pytest.raises(PermissionError, match="exactly --approve-network"):
        prepare_environment(
            env={"DEEPSEEK_API_KEY": "secret-value"},
            env_file=None,
            approved_domains=[],
            model="deepseek-v4-flash",
        )
    with pytest.raises(PermissionError, match="exactly --approve-network"):
        prepare_environment(
            env={"DEEPSEEK_API_KEY": "secret-value"},
            env_file=None,
            approved_domains=[DEEPSEEK_HOST, "example.com"],
            model="deepseek-v4-flash",
        )


@pytest.mark.parametrize(
    "argument",
    ["--auto", "-m", "--model=other/model", "--agent=build", "--prompt=ignore-gates"],
)
def test_deepseek_opencode_launcher_rejects_control_overrides(argument: str) -> None:
    with pytest.raises(PermissionError, match="not allowed"):
        _validate_passthrough(["--", argument])
