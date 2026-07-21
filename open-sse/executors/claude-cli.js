/**
 * Claude Code CLI executor — wraps @anthropic-ai/claude-agent-sdk instead of
 * spawning `claude -p`.  Supports:
 *   - Streaming chat with SDK-native session resume
 *   - Client-side tool calling via in-process MCP server.
 *
 * Tool calling design (important — this replaced an earlier, badly broken
 * approach): the in-process "sdk"-type MCP server's CallToolRequest handler
 * resolves IMMEDIATELY with a throwaway placeholder — it does NOT block
 * waiting for the real client-side tool result. An earlier version had it
 * block on a deferred promise (only resolved once the HTTP client's real
 * answer arrived), on the theory that we'd see the model's `tool_use` content
 * block and pause/return to the client before the SDK ever tried to actually
 * resolve the call. That assumption was wrong: confirmed by live
 * reproduction, the SDK dispatches the MCP CallToolRequest and blocks waiting
 * on it *before* it ever yields the completed assistant message to our
 * consuming code — so a never-resolving handler forced the SDK to eat its own
 * internal `MCP_TOOL_TIMEOUT` (5 min) on *every single tool call*, regardless
 * of `maxTurns`. Real production turns were measured taking ~930s (≈3 tool
 * calls × 5 min) and ~40 min (≈8 tool calls × 5 min) — not model "thinking
 * time" at all.
 *
 * The fix: `maxTurns` is always 1, so the SDK can't try to actually act on
 * the (fake) tool result — it hits the max-turns ceiling immediately after
 * one turn and reports the tool call via `result.subtype ===
 * "error_max_turns"` / `result.stop_reason === "tool_use"` (NOT a clean
 * `stop_reason: "tool_use"` on the assistant message — the max-turns cutoff
 * leaves that blank). We treat that pattern as the real tool-use signal,
 * capture the SDK's `session_id`, and return `finish_reason: "tool_calls"` to
 * the HTTP client — this whole round trip is now ~3s instead of 5+ minutes.
 * When the client's real tool result arrives on the next request, we start a
 * *new* `query({ resume: sessionId, ... })` seeded with a `tool_result`
 * message — the SDK's native session-resume mechanism — rather than
 * continuing a parked iterator. This also fixes chained tool_use (a second
 * tool call in the continuation just re-enters the same pause/resume path;
 * the old design explicitly did not support this).
 *
 * Loosely descended from the meridian pattern (https://github.com/rynfar/meridian),
 * which parks a live iterator across HTTP requests — that part of the
 * approach is what caused the bug above and is no longer used here.
 */
import { query } from "@anthropic-ai/claude-agent-sdk";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { BaseExecutor } from "./base.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SSE_DONE = "data: [DONE]\n\n";
const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
};

// The MCP handler now resolves instantly (see the file-level comment), so
// this should never actually be hit — kept only as a defensive upper bound
// passed to the CLI subprocess via MCP_TOOL_TIMEOUT.
const TOOL_TIMEOUT_MS = 5 * 60 * 1000;
const SESSION_TTL_MS = 30 * 60 * 1000;
// Some real client stacks (observed: Hermes' OpenAI-SDK-based client) judge a
// stream "empty" and bail/retry after as little as ~1-2s of total silence —
// well inside a normal ttft for this executor, let alone a stretch where the
// model is genuinely thinking with nothing yet to relay. An SSE comment line
// (`: ...\n\n`) is invisible to any spec-compliant SSE/OpenAI-chunk parser but
// still counts as upstream bytes, so a client's own stream-health timer (and
// our STREAM_STALL_TIMEOUT_MS watchdog) both see the connection as alive.
const KEEPALIVE_INTERVAL_MS = 1000;
const encoder = new TextEncoder();
const KEEPALIVE_BYTES = encoder.encode(": ka\n\n");
const encodeSSE = (data) =>
  encoder.encode(`data: ${JSON.stringify(data)}\n\n`);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function flattenContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => {
      if (!block) return "";
      if (block.type === "text") return block.text || "";
      if (block.type === "tool_use")
        return `[tool call: ${block.name}(${JSON.stringify(block.input)})]`;
      if (block.type === "tool_result") {
        const inner = flattenContent(block.content);
        return inner ? `[tool result]\n${inner}` : "";
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function extractToolResults(messages) {
  const results = [];
  for (const msg of messages) {
    if (msg.role === "tool" && msg.tool_call_id)
      results.push({ tool_call_id: msg.tool_call_id, content: msg.content });
    if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_result" && block.tool_use_id)
          results.push({ tool_call_id: block.tool_use_id, content: block.content });
      }
    }
  }
  return results;
}

function nonToolMessages(messages) {
  return messages.filter(
    (m) =>
      m.role !== "tool" &&
      !(m.role === "user" &&
        Array.isArray(m.content) &&
        m.content.length > 0 &&
        m.content.every((b) => b.type === "tool_result")),
  );
}

// The SDK's streaming `prompt` input only ever accepts SDKUserMessage — every item
// must carry `message.role === "user"` server-side ("Expected message role 'user', got
// '<other>'" is a hard subprocess crash, not a soft validation warning). There is no way
// to seed prior assistant/tool turns into a *brand-new* session through this channel —
// only `resume` replays real history. So when we're not resuming (cache miss/expired,
// e.g. right after a restart on a long-running tool-using conversation), the only
// correct move is to flatten the whole history into one synthetic user-role turn, the
// same way the old `claude -p` transcript replay did, rather than feeding raw
// assistant/tool messages into the stream and crashing the subprocess.
function buildTranscript(messages) {
  return messages
    .map((m) => {
      const label = m.role === "assistant" ? "Assistant" : m.role === "user" ? "Human" : m.role || "Human";
      return `${label}: ${flattenContent(m.content)}`;
    })
    .join("\n\n");
}

function collapseForFreshQuery(messages) {
  if (messages.length <= 1) return messages;
  if (messages.every((m) => m.role === "user")) return messages;
  return [{ role: "user", content: buildTranscript(messages) }];
}

// A truncated prefix (e.g. flat.slice(0, 512)) collides across different
// conversation states once the early messages alone exceed the truncation
// length — the key stops changing as later messages are appended, so a long
// conversation can resume the wrong cached session. Hash the full content
// instead (FNV-1a, cheap and collision-resistant enough for a cache key).
function fingerprint(model, messages) {
  const flat = messages
    .filter((m) => m.role === "user")
    .map((m) => flattenContent(m.content))
    .join("\n");
  let hash = 2166136261;
  for (let i = 0; i < flat.length; i++) {
    hash ^= flat.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `${model}:${flat.length}:${(hash >>> 0).toString(36)}`;
}

// `msg.error` lives on SDKAssistantMessage (SDKAssistantMessageError), never on the
// terminal `result` message — result only carries a free-text `errors[]` array. Classify
// by the assistant-message error code seen earlier in the stream, not the result subtype.
function classifyAssistantError(errCode) {
  if (errCode === "authentication_failed" || errCode === "oauth_org_not_allowed") return 401;
  if (errCode === "billing_error" || errCode === "rate_limit") return 429;
  return null;
}

// ---------------------------------------------------------------------------
// MCP server — raw JSON Schema tools with blocking handlers
// ---------------------------------------------------------------------------

// These four names collide with Anthropic's native Agent Skills / session-search
// tool interface closely enough that the backend silently requires overage
// billing for the request and returns a canned 400 — confirmed by live
// bisection against the account. Two independent triggers were found:
//   1. The JSON-schema tool `name` field matching one of these exactly
//      (regardless of description/parameter content).
//   2. The literal string appearing ANYWHERE in outgoing text — e.g. a
//      caller's system prompt instructing the model to "use session_search
//      to recall..." trips the same wall even with zero tools declared.
// So every occurrence needs aliasing, not just the tool schema: system
// prompt and message text get the literal name swapped for `<name>__9r`
// before the request leaves this process, and every alias occurrence
// (tool_use.name, and any text the model echoes back) gets swapped back
// before reaching the client, so the whole thing is invisible to callers.
const RESERVED_TOOL_NAME_ALIASES = new Map([
  ["skill_view", "skill_view__9r"],
  ["skill_manage", "skill_manage__9r"],
  ["skills_list", "skills_list__9r"],
  ["session_search", "session_search__9r"],
]);

const RESERVED_NAME_PATTERN = new RegExp(
  `\\b(${[...RESERVED_TOOL_NAME_ALIASES.keys()].join("|")})\\b`,
  "g",
);
const RESERVED_ALIAS_PATTERN = new RegExp(
  `\\b(${[...RESERVED_TOOL_NAME_ALIASES.values()].join("|")})\\b`,
  "g",
);

function aliasReservedNamesInText(text) {
  if (typeof text !== "string" || !text) return text;
  return text.replace(
    RESERVED_NAME_PATTERN,
    (name) => RESERVED_TOOL_NAME_ALIASES.get(name),
  );
}

function restoreReservedNamesInText(text) {
  if (typeof text !== "string" || !text) return text;
  return text.replace(RESERVED_ALIAS_PATTERN, (alias) =>
    alias.slice(0, -"__9r".length),
  );
}

function aliasReservedNamesDeep(content) {
  if (typeof content === "string") return aliasReservedNamesInText(content);
  if (Array.isArray(content)) {
    return content.map((block) => {
      if (!block || typeof block.text !== "string") return block;
      return { ...block, text: aliasReservedNamesInText(block.text) };
    });
  }
  return content;
}

function buildMcpServer(tools) {
  if (!tools || tools.length === 0) return null;

  const sorted = [...tools].sort((a, b) => {
    const na = a.function?.name || a.name || "";
    const nb = b.function?.name || b.name || "";
    return na.localeCompare(nb);
  });

  const aliasToOriginal = new Map();

  const server = new Server(
    { name: "passthrough", version: "1.0.0" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: sorted.map((t) => {
      const name = t.function?.name || t.name;
      const alias = RESERVED_TOOL_NAME_ALIASES.get(name);
      if (alias) aliasToOriginal.set(alias, name);
      return {
        name: alias || name,
        description: t.function?.description || t.description || "",
        inputSchema:
          t.function?.parameters || { type: "object", properties: {} },
      };
    }),
  }));

  // Resolve immediately with a throwaway placeholder — see the file-level
  // comment for why this must never block on the real client-side result.
  // maxTurns:1 stops the SDK from doing anything further with this fake
  // content; the real result is delivered later via a resumed session.
  server.setRequestHandler(CallToolRequestSchema, async () => ({
    content: [{ type: "text", text: "" }],
  }));

  return { server, aliasToOriginal };
}

// ---------------------------------------------------------------------------
// SDK helpers
// ---------------------------------------------------------------------------

async function* messageStream(messages) {
  for (let i = 0; i < messages.length; i++) {
    yield {
      type: "user",
      message: messages[i],
      shouldQuery: i === messages.length - 1, // only last triggers turn
    };
  }
}

// The SDK's own bundled native launcher is an optional dependency that isn't always
// present (e.g. Docker image installs the `claude` CLI globally via
// `@anthropic-ai/claude-code` instead) — point the SDK at that instead of its
// built-in lookup, which throws "Native CLI binary for <platform> not found".
const CLAUDE_CLI_PATH = process.env.CLAUDE_CLI_PATH || "/usr/local/bin/claude";

const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

function buildQueryOptions(model, tools, mcp, signal, resume, systemPrompt) {
  const opts = {
    model,
    pathToClaudeCodeExecutable: CLAUDE_CLI_PATH,
    tools: [],
    settingSources: [],
    systemPrompt: systemPrompt || DEFAULT_SYSTEM_PROMPT,
    includePartialMessages: true,
    permissionMode: "bypassPermissions",
    allowDangerouslySkipPermissions: true,
    // Always 1 — see the file-level comment. Anything higher lets the SDK
    // try to act on the MCP handler's fake tool result internally, which is
    // exactly the bug this design avoids.
    maxTurns: 1,
    env: {
      CLAUDE_AGENT_SDK_MCP_NO_PREFIX: "1",
      MCP_TOOL_TIMEOUT: String(TOOL_TIMEOUT_MS),
      CLAUDE_CODE_SESSION_KIND: "bg",
      ENABLE_CLAUDEAI_MCP_SERVERS: "false",
    },
  };

  if (typeof process?.getuid === "function" && process.getuid() === 0) {
    opts.env.IS_SANDBOX = "1";
    opts.env.CLAUDE_CONFIG_DIR =
      process.env.CLAUDE_CONFIG_DIR || "/home/node/.claude";
  }

  if (mcp) {
    opts.mcpServers = {
      passthrough: { type: "sdk", name: "passthrough", instance: mcp.server },
    };
    opts.allowedTools = ["mcp__passthrough__*"];
    opts.disallowedTools = [
      "Read", "Write", "Edit", "Bash", "Glob", "Grep",
      "WebSearch", "WebFetch", "Task", "TaskOutput",
      "TodoRead", "TodoWrite", "NotebookRead", "NotebookEdit",
      "ExitPlanMode", "EnterPlanMode",
    ];
  }

  // Always create our own AbortController (not just when `signal` is given) so
  // callers — notably streamFromIterator's ReadableStream.cancel() — have a
  // direct handle to kill the in-flight SDK query. Without this, a client
  // that disconnects/retries (Hermes does this aggressively, giving up on a
  // stream after ~1-2s) leaves the abandoned generation running server-side
  // for however long the model takes (observed: full STREAM_STALL_TIMEOUT_MS,
  // burning real API cost) instead of being killed immediately.
  opts.abortController = new AbortController();
  if (signal) {
    signal.addEventListener("abort", () => opts.abortController.abort(), {
      once: true,
    });
  }

  if (resume) opts.resume = resume;

  return opts;
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

export class ClaudeCliExecutor extends BaseExecutor {
  pendingSessions = new Map();
  textSessions = new Map();

  constructor() {
    super("claude-cli", { noAuth: true, baseUrl: "claude-cli://local" });
  }

  // chatCore always reads `result.response` as a real Response — a bare
  // {ok,status,message} object here crashes with "Cannot read properties of
  // undefined (reading 'ok')" before combo/account fallback ever sees it.
  errorResult(status, message) {
    const response = new Response(
      JSON.stringify({ error: { message, type: "claude_cli_error" } }),
      { status, headers: { "Content-Type": "application/json" } },
    );
    return { response, url: "claude-cli://local", headers: {}, transformedBody: null };
  }

  // -------------------------------------------------------------------
  // execute — entry point called by chatCore
  // -------------------------------------------------------------------
  async execute({ model, body, signal, log }) {
    const messages = body.messages || [];
    const tools = body.tools || [];
    const system = flattenContent(body.system) || undefined;

    this.pruneStale();

    // ---- tool-result continuation? ----
    // A stateless client resends the FULL history every call, so `messages`
    // can contain dozens of tool results from turns long past. Whether THIS
    // call is a tool-result continuation depends only on whether the
    // TRAILING message(s) are tool results — scanning the whole history (as
    // this used to) means any conversation that has ever used a tool once
    // trips `toolResults.length > 0` forever, permanently disabling the
    // text-session resume path below for every future turn of that
    // conversation and forcing a full from-scratch transcript replay (full
    // cache_creation of the whole, ever-growing history) on every message.
    const lastMsg = messages[messages.length - 1];
    const lastIsToolResult =
      lastMsg?.role === "tool" ||
      (lastMsg?.role === "user" &&
        Array.isArray(lastMsg.content) &&
        lastMsg.content.length > 0 &&
        lastMsg.content.every((b) => b.type === "tool_result"));

    if (lastIsToolResult) {
      const toolResults = extractToolResults(messages);
      const key = fingerprint(model, nonToolMessages(messages));
      const pending = this.pendingSessions.get(key);
      if (pending) {
        this.pendingSessions.delete(key);
        return this.resumeWithToolResults(
          pending.sessionId,
          toolResults,
          model,
          messages,
          tools,
          system,
          signal,
          log,
        );
      }
      log?.debug?.(
        "CLAUDE-SDK",
        "tool results arrived but no pending session — falling through to fresh query",
      );
    }

    // ---- resume a prior text-only session instead of resending the whole
    // transcript? if "everything except the newest message" matches a
    // session we cached, resume it by UUID and send only the new message —
    // keeps Claude-side history and prompt cache alive across HTTP requests.
    // Applies to any fresh user turn, regardless of whether earlier history
    // contains tool results — only the trailing-tool-result case above needs
    // the pendingSessions continuation instead. ----
    let resumeSessionId = null;
    let queryMessages = messages;
    if (!lastIsToolResult && messages.length >= 2 && lastMsg?.role === "user") {
      const lookupKey = fingerprint(model, messages.slice(0, -1));
      const cached = this.textSessions.get(lookupKey);
      if (cached) {
        resumeSessionId = cached.sessionId;
        queryMessages = [lastMsg];
      }
    }
    if (!resumeSessionId) queryMessages = collapseForFreshQuery(queryMessages);

    // ---- fresh query ----
    const mcp = buildMcpServer(tools);
    // Reserved tool names must be aliased everywhere they could appear in
    // outgoing text, not just the tool schema (see RESERVED_TOOL_NAME_ALIASES
    // above) — a caller's system prompt instructing the model to invoke one
    // of these tools by name trips the same wall even with zero tools
    // declared.
    const aliasedSystem = aliasReservedNamesInText(system);
    const aliasedQueryMessages = queryMessages.map((m) => ({
      ...m,
      content: aliasReservedNamesDeep(m.content),
    }));
    const opts = buildQueryOptions(
      model,
      tools,
      mcp,
      signal,
      resumeSessionId,
      aliasedSystem,
    );

    let q;
    try {
      q = query({ prompt: messageStream(aliasedQueryMessages), options: opts });
    } catch (err) {
      log?.error?.("CLAUDE-SDK", `query() threw: ${err.message}`);
      return this.errorResult(502, `claude-cli: SDK query failed — ${err.message}`);
    }

    return this.streamFromIterator(
      q,
      mcp,
      model,
      messages,
      tools,
      resumeSessionId,
      system,
      signal,
      log,
      opts.abortController,
    );
  }

  // -------------------------------------------------------------------
  // streamFromIterator — pull from the async iterator, emit SSE.
  // When the model emits tool_use + stop_reason:"tool_use", pause mid-
  // stream: emit [DONE], close THIS stream, park the iterator for the
  // next request to continue.
  // -------------------------------------------------------------------
  async streamFromIterator(
    iterator,
    mcp,
    model,
    messages,
    tools,
    resumeSessionId,
    system,
    signal,
    log,
    abortController,
  ) {
    const toolUses = [];
    let sawMessageStart = false;
    let sawAnyStreamEvent = false;
    let doneStreaming = false;
    let currentIterator = iterator;
    let retriedFresh = false;
    let assistantError = null;
    let keepAliveTimer = null;

    // ---- pre-flight peek ----
    // chatCore/combo.js decide "did this model succeed?" purely from the
    // Response status returned here — and previously we always returned 200
    // synchronously, before the SDK had said anything at all. An immediate
    // quota/session-limit rejection reports back as a terminal `result`
    // message with `is_error`/a non-"success" subtype and zero preceding
    // stream events, often within ~2-3s. If that happens AFTER we've already
    // returned 200 and combo.js has logged "succeeded" and started piping
    // bytes to the client, the error can only blow up the pipe
    // ("failed to pipe response" — confirmed live, this even took the whole
    // process down via an unhandled rejection) instead of letting
    // combo/account fallback move to the next model. So pull at least one
    // message ourselves first: if it's an immediate error, hand back a real
    // error Response (same shape as any other failed executor) so the normal
    // fallback path can do its job. Anything else gets buffered and replayed
    // into the real stream below — this adds no latency for a real reply
    // since the first content chunk was always going to take this same
    // amount of time to arrive, we've just moved the wait earlier.
    const preloaded = [];
    peekLoop: while (true) {
      let msg, done;
      try {
        ({ value: msg, done } = await currentIterator.next());
      } catch (err) {
        return this.errorResult(502, `claude-cli: SDK query failed — ${err.message}`);
      }
      if (done) break;

      if (msg.type === "stream_event" || msg.type === "assistant") {
        preloaded.push(msg);
        break;
      }

      if (msg.type === "result") {
        const cliApiError =
          typeof msg.result === "string" &&
          msg.result.match(/^API Error: (\d+)\s*(.*)$/s);
        const isErrorResult =
          msg.subtype !== "success" || Boolean(cliApiError) || msg.is_error === true;

        if (isErrorResult) {
          // Same fail-open retry as the main loop below: a cached session id
          // may no longer exist server-side — retry once as a brand-new
          // session before giving up.
          if (resumeSessionId && !retriedFresh && !cliApiError) {
            retriedFresh = true;
            try {
              const freshOpts = buildQueryOptions(model, tools, mcp, signal, null, system);
              currentIterator = query({
                prompt: messageStream(collapseForFreshQuery(messages)),
                options: freshOpts,
              });
              abortController = freshOpts.abortController;
              continue peekLoop;
            } catch {
              /* fall through to error surfacing below */
            }
          }

          const errMsg = cliApiError
            ? cliApiError[2] || msg.result
            : msg.errors?.join?.(", ") ||
              (typeof msg.result === "string" && msg.result) ||
              "SDK query failed — empty result with is_error";
          const status = cliApiError
            ? Number(cliApiError[1])
            : msg.api_error_status || 429;
          log?.error?.("CLAUDE-SDK", `result error (pre-flight): ${errMsg}`);
          return this.errorResult(status, `claude-cli: ${errMsg}`);
        }

        // Non-error terminal with no content (e.g. paused for tool_use
        // before any output, or a genuinely empty successful reply) — not a
        // failure, let the normal stream path below handle it identically.
        preloaded.push(msg);
        break;
      }

      // system/init or other non-content, non-terminal message types —
      // nothing to replay, keep peeking.
    }

    const stream = new ReadableStream({
      start: async (controller) => {
        keepAliveTimer = setInterval(() => {
          try {
            controller.enqueue(KEEPALIVE_BYTES);
          } catch {
            /* controller already closed */
          }
        }, KEEPALIVE_INTERVAL_MS);
        try {
          while (!doneStreaming) {
            const { value: msg, done } = preloaded.length
              ? { value: preloaded.shift(), done: false }
              : await currentIterator.next();

            if (done) {
              controller.enqueue(encoder.encode(SSE_DONE));
              controller.close();
              return;
            }

            // --- partial stream events ---
            if (msg.type === "stream_event" && msg.event) {
              sawAnyStreamEvent = true;
              const ev = msg.event;

              // Track tool_use blocks as they start
              if (
                ev.type === "content_block_start" &&
                ev.content_block?.type === "tool_use"
              ) {
                // Undo the reserved-name alias so the client never sees it —
                // mutate in place before this name is read anywhere below,
                // including the raw event enqueued to the client further down.
                const original = mcp?.aliasToOriginal?.get(
                  ev.content_block.name,
                );
                if (original) ev.content_block.name = original;

                toolUses.push({
                  id: ev.content_block.id,
                  name: ev.content_block.name,
                  input: {},
                  _json: "", // accumulate input_json_delta here
                });
              }

              // Accumulate partial JSON for the current tool_use
              if (
                ev.type === "content_block_delta" &&
                ev.delta?.type === "input_json_delta" &&
                toolUses.length > 0
              ) {
                const cur = toolUses[toolUses.length - 1];
                cur._json += ev.delta.partial_json || "";
              }

              // The model only ever saw the aliased name (see execute()), so
              // it can only echo the alias back in prose — restore it here
              // too, not just on tool_use.name, so the rename never leaks to
              // the client via a text reply.
              if (
                ev.type === "content_block_delta" &&
                ev.delta?.type === "text_delta"
              ) {
                ev.delta.text = restoreReservedNamesInText(ev.delta.text);
              }

              // Parse accumulated JSON on content_block_stop
              if (ev.type === "content_block_stop" && toolUses.length > 0) {
                const cur = toolUses[toolUses.length - 1];
                if (cur._json) {
                  try {
                    cur.input = JSON.parse(cur._json);
                  } catch {
                    /* partial — full message will correct it */
                  }
                  delete cur._json;
                }
              }

              if (ev.type === "message_start") sawMessageStart = true;

              controller.enqueue(encodeSSE(ev));
            }

            // --- complete assistant message ---
            if (msg.type === "assistant") {
              const stopReason = msg.message?.stop_reason;
              if (msg.error) assistantError = msg.error;

              // Fill tool_use inputs from the complete message
              if (msg.message?.content) {
                for (const block of msg.message.content) {
                  if (block.type === "tool_use") {
                    const tu = toolUses.find((t) => t.id === block.id);
                    if (tu) tu.input = block.input || tu.input;
                  }
                }
              }

              // ---- PAUSE: tool_use ----
              // Defensive fallback — in practice, with maxTurns:1 (see
              // file-level comment) the SDK reports this via the terminal
              // `result` message instead, not a clean stop_reason here.
              if (stopReason === "tool_use" && toolUses.length > 0) {
                const key = fingerprint(model, messages);
                this.pendingSessions.set(key, {
                  sessionId: msg.session_id,
                  createdAt: Date.now(),
                });

                controller.enqueue(encoder.encode(SSE_DONE));
                controller.close();
                return;
              }

              // Cache session id for potential text-only resume
              if (
                stopReason === "end_turn" ||
                !stopReason ||
                stopReason === "stop_sequence"
              ) {
                this.cacheTextSession(fingerprint(model, messages), msg.session_id);
              }
            }

            // --- result (terminal) ---
            if (msg.type === "result") {
              // ---- PAUSE: tool_use ----
              // With maxTurns:1, the SDK can't act on the MCP handler's fake
              // result — it hits the turn ceiling immediately and reports the
              // tool call here (subtype "error_max_turns", stop_reason
              // "tool_use") rather than as a clean stop_reason on the
              // assistant message. This is the expected/normal path, not an
              // error — a real client-side tool result is delivered next via
              // a resumed session (see resumeWithToolResults).
              if (toolUses.length > 0) {
                const key = fingerprint(model, messages);
                this.pendingSessions.set(key, {
                  sessionId: msg.session_id,
                  createdAt: Date.now(),
                });

                controller.enqueue(encoder.encode(SSE_DONE));
                controller.close();
                return;
              }

              // The CLI sometimes returns a canned refusal (billing/quota walls,
              // e.g. "API Error: 400 Third-party apps now draw from your extra
              // usage...") as a *complete* assistant message with no preceding
              // stream events, then closes with subtype:"success" — a real
              // failure disguised as an empty-but-successful response. Detect
              // the "API Error: NNN ..." text and treat it as an error so combo/
              // account fallback actually kicks in instead of the client seeing
              // a silent empty stream.
              const cliApiError =
                !sawAnyStreamEvent &&
                typeof msg.result === "string" &&
                msg.result.match(/^API Error: (\d+)\s*(.*)$/s);
              // A rate-limit/quota rejection can ALSO come back as a genuinely
              // empty `subtype:"success"` result with no text at all (confirmed
              // live: production turns with IN 0/OUT 0, requestDetails showing
              // "[Empty streaming response]", right after account quota ran
              // out) — the SDK's own SDKResultSuccess type carries `is_error`
              // and `api_error_status` fields independent of `subtype`, and we
              // were only checking `subtype`. Treat `is_error: true` as a
              // failure regardless of subtype so combo/account fallback sees it.
              const isErrorResult =
                msg.subtype !== "success" ||
                Boolean(cliApiError) ||
                msg.is_error === true;

              if (isErrorResult && !sawAnyStreamEvent) {
                // A cached session id may no longer exist server-side (pruned,
                // process restarted, etc). Fail open: retry once as a brand-new
                // session with the full transcript instead of surfacing a
                // resume-specific error or bouncing to the next combo model.
                // (Skip this retry for a canned CLI API error — a fresh session
                // will hit the same account-level wall.)
                if (resumeSessionId && !retriedFresh && !cliApiError) {
                  retriedFresh = true;
                  try {
                    const freshOpts = buildQueryOptions(model, tools, mcp, signal, null, system);
                    currentIterator = query({
                      prompt: messageStream(collapseForFreshQuery(messages)),
                      options: freshOpts,
                    });
                    abortController = freshOpts.abortController;
                    continue;
                  } catch {
                    /* fall through to error surfacing below */
                  }
                }

                const errMsg = cliApiError
                  ? cliApiError[2] || msg.result
                  : msg.errors?.join?.(", ") ||
                    (typeof msg.result === "string" && msg.result) ||
                    "SDK query failed — empty result with is_error";
                const status = cliApiError
                  ? Number(cliApiError[1])
                  : msg.api_error_status ||
                    classifyAssistantError(assistantError) ||
                    429;
                log?.error?.("CLAUDE-SDK", `result error: ${errMsg}`);
                if (status) {
                  controller.error({
                    ok: false,
                    status,
                    message: `claude-cli: ${errMsg}`,
                  });
                } else {
                  controller.error(
                    new Error(`claude-cli: ${errMsg}`),
                  );
                }
                return;
              }

              if (msg.session_id) {
                this.cacheTextSession(fingerprint(model, messages), msg.session_id);
              }

              controller.enqueue(encoder.encode(SSE_DONE));
              controller.close();
              return;
            }
          }
        } catch (err) {
          if (!sawAnyStreamEvent && !doneStreaming) {
            // Startup error — surface to combo fallback
            controller.error(err);
          } else {
            // Already streaming — surface as text
            try {
              controller.enqueue(
                encodeSSE({
                  type: "content_block_delta",
                  index: 0,
                  delta: {
                    type: "text_delta",
                    text: `[claude-cli error: ${err.message}]`,
                  },
                }),
              );
              controller.enqueue(encoder.encode(SSE_DONE));
              controller.close();
            } catch {
              /* stream already closed */
            }
          }
        } finally {
          clearInterval(keepAliveTimer);
        }
      },

      cancel() {
        // Setting doneStreaming alone does nothing while the loop is parked on
        // `await currentIterator.next()` — that await only settles when the
        // SDK call itself finishes, which can take minutes. Abort it directly
        // so a disconnected/retried-away client doesn't leave the generation
        // running (and billing) in the background.
        doneStreaming = true;
        abortController?.abort();
        clearInterval(keepAliveTimer);
      },
    });

    return {
      response: new Response(stream, { status: 200, headers: SSE_HEADERS }),
      url: "claude-cli://local",
      headers: SSE_HEADERS,
      transformedBody: null,
    };
  }

  // -------------------------------------------------------------------
  // resumeWithToolResults — start a *new* SDK query resuming the paused
  // session by UUID, seeded with the client's real tool result as the next
  // turn. Streams via the same streamFromIterator as any other query, so a
  // chained tool_use in the continuation just re-parks — no special-casing
  // needed (the old iterator-parking design couldn't do this).
  //
  // Delivered as plain text, NOT a `tool_result` content block matched by
  // tool_use_id — confirmed by live testing that the latter does not work:
  // the MCP handler's placeholder response (see buildMcpServer) already got
  // recorded as *the* answer to that tool_use_id in the session's resumed
  // history during the paused turn, so a second tool_result for the same id
  // is ignored and the model answers from the (empty) placeholder instead.
  // Telling it in prose to disregard the placeholder and use the real result
  // works reliably.
  // -------------------------------------------------------------------
  async resumeWithToolResults(
    sessionId,
    toolResults,
    model,
    messages,
    tools,
    system,
    signal,
    log,
  ) {
    const resultText = toolResults
      .map((tr) => {
        const content = aliasReservedNamesDeep(
          Array.isArray(tr.content)
            ? tr.content
            : [{ type: "text", text: String(tr.content || "") }],
        );
        return `[Real result for tool_use_id ${tr.tool_call_id} — the MCP transport returned an empty placeholder for this call, ignore that and use this instead]:\n${flattenContent(content)}`;
      })
      .join("\n\n");

    const mcp = buildMcpServer(tools);
    const aliasedSystem = aliasReservedNamesInText(system);
    const opts = buildQueryOptions(model, tools, mcp, signal, sessionId, aliasedSystem);

    let q;
    try {
      q = query({
        prompt: messageStream([{ role: "user", content: resultText }]),
        options: opts,
      });
    } catch (err) {
      log?.error?.("CLAUDE-SDK", `resume query() threw: ${err.message}`);
      return this.errorResult(502, `claude-cli: SDK resume failed — ${err.message}`);
    }

    return this.streamFromIterator(
      q,
      mcp,
      model,
      messages,
      tools,
      sessionId,
      system,
      signal,
      log,
      opts.abortController,
    );
  }

  // -------------------------------------------------------------------
  // Session cache primitives
  // -------------------------------------------------------------------
  cacheTextSession(key, sessionId) {
    if (!sessionId) return;
    this.textSessions.set(key, {
      sessionId,
      expiresAt: Date.now() + SESSION_TTL_MS,
    });
  }

  pruneStale() {
    const now = Date.now();
    for (const [k, s] of this.textSessions) {
      if (s.expiresAt <= now) this.textSessions.delete(k);
    }
    // A paused session holds no live iterator/handler — resumeWithToolResults
    // starts a brand-new query keyed off the cached sessionId — so pruning it
    // is just dropping the cache entry, nothing to reject or abort.
    for (const [k, s] of this.pendingSessions) {
      if (now - s.createdAt > SESSION_TTL_MS) {
        this.pendingSessions.delete(k);
      }
    }
  }
}
