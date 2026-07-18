import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { BaseExecutor } from "./base.js";
import { PROVIDERS } from "../config/providers.js";
import { SSE_HEADERS, SSE_DONE } from "../utils/sseConstants.js";

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
  const system = flattenContent(body.system);
  if (system) args.push("--system-prompt", system);
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
    let buffer = "";

    const body_ = new ReadableStream({
      start(controller) {
        child.stdout.on("data", chunk => {
          buffer += chunk.toString("utf8");
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";
          for (const line of lines) emitLine(line, controller, encoder, state);
        });
        child.stdout.on("end", () => {
          if (buffer.trim()) emitLine(buffer, controller, encoder, state);
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
