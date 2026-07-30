import type { Plugin } from "@opencode-ai/plugin"

const PROTOCOL_REVISION = "2"
const SCHEMA_VERSION = "1.0"

const permissionedTools = new Set([
  "protbind_fetch_public_data",
  "protbind_case_create",
  "protbind_case_advance",
  "protbind_case_attach_support",
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
])

const disclosures: Record<string, {
  reads: string[]
  writes: string[]
  network: string
  scientificStateChange: boolean
}> = {
  protbind_fetch_public_data: {
    reads: ["public registry identifier"],
    writes: ["project-local candidate and provenance receipt"],
    network: "exact approved registry domain",
    scientificStateChange: false,
  },
  protbind_case_create: {
    reads: ["project-local case and frozen index"],
    writes: ["new run manifest and imported input artifacts"],
    network: "none",
    scientificStateChange: true,
  },
  protbind_case_advance: {
    reads: ["current manifest and upstream artifacts"],
    writes: ["one stage record and acceptance receipt"],
    network: "none",
    scientificStateChange: true,
  },
  protbind_case_attach_support: {
    reads: ["one reviewed project-local support artifact"],
    writes: ["content-addressed support and manifest binding"],
    network: "none",
    scientificStateChange: true,
  },
}

type IdleReceipt = {
  task: string
  status: "COMPLETED" | "CANCELLED"
  output_sha256: string | null
}

export type OpenCodeShadowPlan = {
  schema_version: string
  protocol_revision: string
  kind: "protbind.shadow-plan"
  status: string
  plan_id: string
  tool: string
  arguments_sha256: string
  safe_idle_tasks: string[]
  forbidden_before_approval: string[]
  idle_tasks: IdleReceipt[]
}

type PlanState = {
  plan: OpenCodeShadowPlan
  permissionID?: string
  callID?: string
  sessionID: string
  controller: AbortController
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonical).join(",")}]`
  }
  const object = value as Record<string, unknown>
  return `{${Object.keys(object).sort().map((key) =>
    `${JSON.stringify(key)}:${canonical(object[key])}`
  ).join(",")}}`
}

async function sha256(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(canonical(value))
  const digest = await crypto.subtle.digest("SHA-256", bytes)
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("")
}

function disclosure(tool: string) {
  return disclosures[tool] ?? {
    reads: ["declared private data or bounded tool inputs"],
    writes: ["declared bounded tool output, if any"],
    network: "none unless the native permission explicitly discloses it",
    scientificStateChange: false,
  }
}

export async function buildOpenCodeShadowPlan(
  tool: string,
  argumentMaterial: unknown,
): Promise<OpenCodeShadowPlan> {
  const argumentsSha256 = await sha256(argumentMaterial)
  const safeIdleTasks = [
    "render-action-preview",
    "compile-conditional-branches",
    "prepare-cancellable-report-skeleton",
  ]
  const info = disclosure(tool)
  if (info.network !== "none") {
    safeIdleTasks.push("render-exact-network-disclosure")
  }
  if (info.writes.length > 0) {
    safeIdleTasks.push("render-declared-write-set")
  }
  if (info.scientificStateChange) {
    safeIdleTasks.push("compile-one-stage-postflight-checklist")
  }
  const safe = {
    schema_version: SCHEMA_VERSION,
    protocol_revision: PROTOCOL_REVISION,
    kind: "protbind.shadow-plan" as const,
    status: "WAITING_APPROVAL",
    tool,
    arguments_sha256: argumentsSha256,
    safe_idle_tasks: safeIdleTasks,
    forbidden_before_approval: [
      "private-data-read",
      "network-access",
      "scientific-state-write",
      "continuation-token-use",
      "memory-write",
    ],
    idle_tasks: [] as IdleReceipt[],
  }
  return {
    ...safe,
    plan_id: await sha256(safe),
  }
}

function toolFromPermission(permission: Record<string, unknown>): string | undefined {
  const metadata = (
    permission.metadata && typeof permission.metadata === "object"
      ? permission.metadata
      : {}
  ) as Record<string, unknown>
  const candidates = [
    metadata.tool,
    metadata.toolID,
    metadata.name,
    permission.type,
  ]
  return candidates.find(
    (value): value is string =>
      typeof value === "string" && permissionedTools.has(value),
  )
}

function argumentMaterial(permission: Record<string, unknown>): unknown {
  const metadata = (
    permission.metadata && typeof permission.metadata === "object"
      ? permission.metadata
      : {}
  ) as Record<string, unknown>
  return metadata.args ?? metadata.arguments ?? {
    pattern: permission.pattern ?? null,
    callID: permission.callID ?? null,
  }
}

async function runIdleTasks(state: PlanState): Promise<void> {
  for (const task of state.plan.safe_idle_tasks) {
    await new Promise((resolve) => setTimeout(resolve, 0))
    if (state.controller.signal.aborted) {
      state.plan.idle_tasks.push({
        task,
        status: "CANCELLED",
        output_sha256: null,
      })
      continue
    }
    state.plan.idle_tasks.push({
      task,
      status: "COMPLETED",
      output_sha256: await sha256({
        plan_id: state.plan.plan_id,
        task,
        scientific_result: "pending",
      }),
    })
  }
}

export const ProtBindShadowPlan: Plugin = async ({ client, directory }) => {
  const byPermission = new Map<string, PlanState>()
  const byCall = new Map<string, PlanState>()

  const findState = (sessionID: string, tool: string, callID: string) => {
    const exact = byCall.get(callID)
    if (exact) return exact
    const fallback = Array.from(byPermission.values()).reverse().find(
      (state) =>
        state.sessionID === sessionID &&
        state.plan.tool === tool &&
        state.plan.status === "APPROVED",
    )
    if (fallback) {
      fallback.callID = callID
      byCall.set(callID, fallback)
    }
    return fallback
  }

  const toast = async (
    message: string,
    variant: "info" | "success" | "warning" | "error" = "info",
  ) => {
    await client.tui.showToast({
      body: {
        title: "ProtBind ShadowPlan",
        message,
        variant,
        duration: 8000,
      },
      query: { directory },
    }).catch(() => undefined)
  }

  const log = async (state: PlanState, message: string) => {
    await client.app.log({
      body: {
        service: "protbind-shadowplan",
        level: "info",
        message,
        extra: {
          sessionID: state.sessionID,
          permissionID: state.permissionID,
          callID: state.callID,
          shadowPlan: state.plan,
        },
      },
      query: { directory },
    }).catch(() => undefined)
  }

  const register = async (permission: Record<string, unknown>) => {
    const permissionID = permission.id
    const sessionID = permission.sessionID
    if (typeof permissionID !== "string" || typeof sessionID !== "string") return
    if (byPermission.has(permissionID)) return
    const tool = toolFromPermission(permission)
    if (!tool) return
    const plan = await buildOpenCodeShadowPlan(tool, argumentMaterial(permission))
    const state: PlanState = {
      plan,
      permissionID,
      callID: typeof permission.callID === "string" ? permission.callID : undefined,
      sessionID,
      controller: new AbortController(),
    }
    byPermission.set(permissionID, state)
    if (state.callID) byCall.set(state.callID, state)
    void runIdleTasks(state)
    await toast(
      `${tool} is WAITING_APPROVAL · plan ${plan.plan_id.slice(0, 12)} · ` +
      `no private read/network/state write has started`,
    )
    await log(state, "permission waiting with redacted deterministic ShadowPlan")
  }

  return {
    "permission.ask": async (input, _output) => {
      await register(input as unknown as Record<string, unknown>)
    },
    event: async ({ event: rawEvent }) => {
      const event = rawEvent as unknown as {
        type: string
        properties: Record<string, unknown>
      }
      if (event.type === "permission.updated" || event.type === "permission.asked") {
        await register(event.properties)
        return
      }
      if (event.type === "permission.replied") {
        const permissionID = event.properties.permissionID
        if (typeof permissionID !== "string") return
        const state = byPermission.get(permissionID)
        if (!state) return
        state.controller.abort()
        const response = String(event.properties.response ?? "")
        const approved = response !== "reject" && response !== "deny"
        state.plan.status = approved ? "APPROVED" : "DECLINED"
        await toast(
          `${state.plan.tool} ${state.plan.status} · plan ${state.plan.plan_id.slice(0, 12)}`,
          approved ? "success" : "warning",
        )
        await log(state, "native OpenCode permission reply observed")
        return
      }
      if (event.type === "session.status") {
        const sessionID = event.properties.sessionID
        const status = event.properties.status as { type?: string } | undefined
        if (typeof sessionID === "string" && status?.type === "idle") {
          const waiting = Array.from(byPermission.values()).filter(
            (state) =>
              state.sessionID === sessionID &&
              state.plan.status === "WAITING_APPROVAL",
          ).length
          if (waiting > 0) {
            await toast(`${waiting} ProtBind approval request(s) still waiting`)
          }
        }
        return
      }
      if (event.type === "session.idle") {
        const sessionID = event.properties.sessionID
        if (typeof sessionID !== "string") return
        const waiting = Array.from(byPermission.values()).filter(
          (state) =>
            state.sessionID === sessionID &&
            state.plan.status === "WAITING_APPROVAL",
        ).length
        if (waiting > 0) {
          await toast(`${waiting} ShadowPlan(s) remain conditional; no science was advanced`)
        }
      }
    },
    "tool.execute.before": async (input, _output) => {
      if (!permissionedTools.has(input.tool)) return
      const state = findState(input.sessionID, input.tool, input.callID)
      if (!state) return
      state.controller.abort()
      state.plan.status = "DISPATCHING"
      await log(state, "approved native tool is dispatching")
    },
    "tool.execute.after": async (input, output) => {
      if (!permissionedTools.has(input.tool)) return
      const state = findState(input.sessionID, input.tool, input.callID)
      if (!state) return
      state.plan.status = "EXECUTED"
      output.title = `${output.title} · ShadowPlan ${state.plan.plan_id.slice(0, 12)} adopted`
      output.metadata = {
        ...(output.metadata ?? {}),
        protbind_shadow_plan: {
          protocol_revision: PROTOCOL_REVISION,
          plan_id: state.plan.plan_id,
          status: state.plan.status,
          arguments_sha256: state.plan.arguments_sha256,
          idle_tasks: state.plan.idle_tasks,
        },
      }
      await toast(
        `${state.plan.tool} EXECUTED · plan ${state.plan.plan_id.slice(0, 12)}`,
        "success",
      )
      await log(state, "native tool completed with adopted ShadowPlan")
    },
  }
}
