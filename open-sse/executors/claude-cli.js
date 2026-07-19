import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { BaseExecutor } from "./base.js";
import { PROVIDERS } from "../config/providers.js";
import { SSE_HEADERS, SSE_DONE } from "../utils/sseConstants.js";
import { HTTP_STATUS } from "../config/runtimeConfig.js";

// Same phrase classes the `claude` CLI itself checks for internally (extracted from its
// own bundled auth-error regex) — reused here so chatCore's combo/account fallback
// treats these the same as any other provider's auth/quota failure (see chatCore.js's
// `!providerResponse.ok` branch), instead of silently swallowing them into a 200
// response with the error baked into the assistant's reply text.
const AUTH_ERROR_PATTERN = /not logged in|please run \/login|authentication failed|failed to authenticate|invalid api key|oauth (?:token|session) (?:expired|revoked)|401 unauthorized|403 forbidden|token (?:has )?expired|bad credentials/i;
// Quota/billing exhaustion — distinct from auth so it maps to 429 (rate limited) rather
// than 401, giving it accountFallback's backoff/retry-after handling instead of a flat lock.
const QUOTA_ERROR_PATTERN = /usage limit reached|credit balance (?:is )?too low/i;

const CLAUDE_BIN = process.env.CLAUDE_CLI_PATH || "claude";

// Flatten an Anthropic content value (string, or array of content blocks) down to
// plain text — `claude -p` takes a single natural-language prompt, not structured
// message content, so images/tool blocks can't round-trip; they're summarized instead.
function flattenContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map(block => {
      if (!block) return "";
      if (block.type === "text") return block.text || "";
      if (block.type === "tool_use") return `[tool call: ${block.name}(${JSON.stringify(block.input)})]`;
      if (block.type === "tool_result") {
        const inner = flattenContent(block.content);
        return inner ? `[tool result]\n${inner}` : "";
      }
      if (block.type === "image") return "[image omitted — claude-cli is text-only]";
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

// `claude -p` is a single-shot, stateless invocation — there's no per-request session
// to hang multi-turn history off, so prior turns are replayed as a plain transcript
// in the prompt itself (same statelessness contract 9Router already expects: clients
// resend full message history on every request).
function buildTranscript(messages = []) {
  return messages
    .map(m => `${m.role === "assistant" ? "Assistant" : "Human"}: ${flattenContent(m.content)}`)
    .join("\n\n");
}

// `claude -p` keeps Claude Code's full agentic persona (skills, memory-recall instincts,
// tool-use reflexes) baked in regardless of --tools — leaving the default system prompt
// in place (i.e. not passing --system-prompt at all) reproducibly caused the model to
// narrate/hallucinate fake tool-call syntax as plain text once real tools were stripped,
// and to otherwise behave like an interactive coding agent rather than a plain assistant
// answering one request. Passing --system-prompt replaces the default wholesale, which
// fixes this; the caller's own system message (if any) is appended after our baseline.
const BASE_SYSTEM_PROMPT = "You are a helpful assistant. Answer directly and concisely. You have no tools available in this session — do not attempt, narrate, or simulate any tool calls.";

function buildArgs(model, body) {
  const args = [
    "-p", "--output-format", "stream-json", "--include-partial-messages", "--verbose",
    "--model", model,
    // No tool access and no persisted session/state for a stateless API call — this
    // executor is a text-completion backend, not an agentic coding session.
    "--tools", "",
    "--no-session-persistence",
    "--setting-sources", "user",
  ];
  const callerSystem = flattenContent(body.system);
  const system = callerSystem ? `${BASE_SYSTEM_PROMPT}\n\n${callerSystem}` : BASE_SYSTEM_PROMPT;
  args.push("--system-prompt", system);
  return args;
}

// Re-shape one `claude -p --output-format stream-json` NDJSON line into SSE `data:`
// frames. `{"type":"stream_event","event":{...}}` lines carry the *exact* raw
// Anthropic Messages-API stream event (message_start/content_block_delta/etc,
// verified against a live `claude -p` run) — re-emitting `event` as-is lets the
// existing claude→openai response translator consume it unmodified.
function emitLine(line, controller, encoder, state) {
  const trimmed = line.trim();
  if (!trimmed) return;
  let parsed;
  try { parsed = JSON.parse(trimmed); } catch { return; }

  if (parsed.type === "stream_event" && parsed.event) {
    controller.enqueue(encoder.encode(`data: ${JSON.stringify(parsed.event)}\n\n`));
    return;
  }
  // CLI-level failure (e.g. auth/session error) with no assistant turn at all —
  // surface it as a minimal text response instead of silently emitting nothing.
  if (parsed.type === "result" && parsed.is_error && !state.sawMessageStart) {
    const msg = parsed.result || "claude CLI reported an error with no response";
    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "message_start", message: { id: `msg_${Date.now()}`, model: state.model, type: "message", role: "assistant", content: [], usage: {} } })}\n\n`));
    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`));
    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: `[claude-cli error] ${msg}` } })}\n\n`));
    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`));
    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: {} })}\n\n`));
    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "message_stop" })}\n\n`));
  }
  if (parsed.type === "stream_event" && parsed.event?.type === "message_start") state.sawMessageStart = true;
}

export class ClaudeCliExecutor extends BaseExecutor {
  constructor() {
    super("claude-cli", PROVIDERS["claude-cli"]);
  }

  // Executor-specific so parseUpstreamError() picks up the plain message we stuffed
  // into the JSON error body below, instead of stringifying the whole object.
  parseError(response, bodyText) {
    try {
      const parsed = JSON.parse(bodyText);
      if (parsed?.error?.message) return { status: response.status, message: parsed.error.message };
    } catch {}
    return { status: response.status, message: bodyText || `HTTP ${response.status}` };
  }

  // Fully overridden — this provider shells out to a local CLI process instead of
  // making an HTTP call, so none of BaseExecutor's fetch/retry/fallback machinery applies.
  async execute({ model, body, signal, log }) {
    const args = buildArgs(model, body);
    const prompt = buildTranscript(body.messages);

    const child = spawn(CLAUDE_BIN, args, {
      // Neutral cwd so per-repo CLAUDE.md/skills don't leak into every API response.
      cwd: tmpdir(),
      stdio: ["pipe", "pipe", "pipe"],
    });

    await new Promise((resolve, reject) => {
      child.once("spawn", resolve);
      child.once("error", reject);
    });

    child.stdin.write(prompt);
    child.stdin.end();

    let stderrBuf = "";
    child.stderr.on("data", d => { stderrBuf += d.toString(); });

    const onAbort = () => { try { child.kill("SIGTERM"); } catch {} };
    signal?.addEventListener("abort", onAbort, { once: true });

    const encoder = new TextEncoder();
    const state = { model, sawMessageStart: false };

    // Don't commit to a response (and therefore an HTTP status) until we know whether
    // the CLI actually started a turn or failed outright (e.g. "Not logged in" with no
    // assistant content at all). Committing to 200 unconditionally — as an earlier
    // version of this executor did — meant chatCore's `!providerResponse.ok` fallback
    // check never fired, so a dead session silently "succeeded" with the error text
    // baked into the assistant's reply instead of the combo trying the next model.
    let rawBuffer = "";
    const pendingLines = [];
    const outcome = await new Promise((resolve) => {
      let settled = false;
      const settle = (result) => {
        if (settled) return;
        settled = true;
        child.stdout.off("data", onData);
        child.stdout.off("end", onEnd);
        resolve(result);
      };
      const onData = (chunk) => {
        rawBuffer += chunk.toString("utf8");
        const lines = rawBuffer.split("\n");
        rawBuffer = lines.pop() || "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          pendingLines.push(trimmed);
          let parsed;
          try { parsed = JSON.parse(trimmed); } catch { continue; }
          if (parsed.type === "stream_event") return settle({ ok: true });
          if (parsed.type === "result" && parsed.is_error) {
            return settle({ ok: false, message: parsed.result || "claude CLI reported an error with no response" });
          }
        }
      };
      const onEnd = () => settle({ ok: false, message: stderrBuf.trim() || "claude CLI exited with no output" });
      child.stdout.on("data", onData);
      child.stdout.on("end", onEnd);
      child.once("error", (err) => settle({ ok: false, message: err.message || "claude CLI process error" }));
    });

    if (!outcome.ok) {
      onAbort();
      const status = QUOTA_ERROR_PATTERN.test(outcome.message)
        ? HTTP_STATUS.RATE_LIMITED
        : AUTH_ERROR_PATTERN.test(outcome.message)
          ? HTTP_STATUS.UNAUTHORIZED
          : HTTP_STATUS.BAD_GATEWAY;
      const response = new Response(
        JSON.stringify({ error: { message: outcome.message, type: "claude_cli_error" } }),
        { status, headers: { "Content-Type": "application/json" } }
      );
      return { response, url: "claude-cli://local", headers: {}, transformedBody: body };
    }

    const body_ = new ReadableStream({
      start(controller) {
        for (const line of pendingLines) emitLine(line, controller, encoder, state);
        child.stdout.on("data", chunk => {
          rawBuffer += chunk.toString("utf8");
          const lines = rawBuffer.split("\n");
          rawBuffer = lines.pop() || "";
          for (const line of lines) emitLine(line, controller, encoder, state);
        });
        child.stdout.on("end", () => {
          if (rawBuffer.trim()) emitLine(rawBuffer, controller, encoder, state);
          controller.enqueue(encoder.encode(SSE_DONE));
          controller.close();
        });
        child.stdout.on("error", err => controller.error(err));
        child.once("error", err => controller.error(err));
      },
      cancel() { onAbort(); }
    });

    if (log?.debug) {
      child.once("exit", (code) => {
        if (code !== 0) log.debug("CLAUDE-CLI", `exited ${code}${stderrBuf ? `: ${stderrBuf.slice(0, 300)}` : ""}`);
      });
    }

    const response = new Response(body_, { status: 200, headers: SSE_HEADERS });
    return { response, url: "claude-cli://local", headers: SSE_HEADERS, transformedBody: body };
  }
}

export default ClaudeCliExecutor;
