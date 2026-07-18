// Runs the locally installed `claude` CLI in print mode (`claude -p`) as the transport
// instead of an HTTP call — see executors/claude-cli.js. Auth comes from whatever
// account the host's `claude` CLI is already logged into (~/.claude session/keychain),
// not a stored API key, so this provider is noAuth from 9Router's point of view.
export default {
  id: "claude-cli",
  priority: 10,
  alias: "cli",
  uiAlias: "cli",
  display: {
    name: "Claude Code CLI",
    icon: "smart_toy",
    color: "#D97757",
    website: "https://claude.ai",
    notice: {
      text: "Runs the locally installed `claude` CLI (claude -p) using whatever account it's already logged into on the host running 9Router. Requests count against that Claude subscription's own usage limits — this does not add capacity, it lets existing Claude Code CLI usage be routed through 9Router.",
    },
  },
  category: "free",
  noAuth: true,
  transport: {
    format: "claude",
    forceStream: true,
    noAuth: true,
  },
  models: [
    { id: "claude-fable-5", name: "Claude Fable 5" },
    { id: "claude-sonnet-5", name: "Claude Sonnet 5" },
    { id: "claude-opus-4-8", name: "Claude Opus 4.8" },
    { id: "claude-opus-4-7", name: "Claude Opus 4.7" },
    { id: "claude-haiku-4-5-20251001", name: "Claude 4.5 Haiku" },
  ],
};
