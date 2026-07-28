/**
 * Tokens-per-second status bar metric + per-turn observation logger.
 *
 * Two views over the same footer slot:
 *   - At rest: decayed average tok/s for the current model, updated at each
 *     message_end. Authoritative (uses usage.output). Prior samples fade with
 *     a 30-min half-life so long sessions track recent behavior; short
 *     sessions behave like a plain cumulative average.
 *   - During streaming: live instantaneous tok/s, updated ~5×/sec.
 *
 * Counters are bucketed per model, so switching models never blends throughput
 * from different models — the displayed number always describes exactly one
 * model. Cycling back to a previously-used model resumes its own average.
 *
 * Live token counting is provider-dependent:
 *   - Anthropic streams cumulative output_tokens live  -> exact live number.
 *   - OpenAI-compatible providers (zai, openai, ...) emit usage only in the
 *     final chunk -> live number is a char-based estimate, shown with a "≈".
 * The estimate accumulates text_delta + thinking_delta + toolcall_delta
 * chars, divided by a *calibrated* chars-per-token ratio learned from
 * completed turns on this model (Bayesian shrinkage toward a code-leaning
 * prior of 3). Converges to the model's true ratio for your content mix
 * after a few turns — more accurate than a flat constant, and unlike a
 * fixed tokenizer it matches GLM's actual vocabulary. At message_end the
 * authoritative usage.output replaces the estimate for the decayed average,
 * so estimates never pollute the long-run number.
 *
 * Timing starts at the first streamed text/thinking delta (not message_start),
 * so both live and cumulative numbers measure decode throughput and exclude
 * TTFT. Starting at message_start would let TTFT (seconds, with xhigh
 * thinking) dominate the denominator and make the live number dip unstable.
 *
 * Until the live estimate has enough data (~0.3s and ~20 tokens), the
 * decayed rest number stays visible — so a new turn doesn't visibly drop
 * the figure to a noisy low value.
 *
 * Decay is leaky integration on both numerator and denominator
 * (0.5^(Δt/HalfLife) per sample), which is the principled form for a
 * time-decayed rate — unlike per-turn EWMA, it isn't biased by variable
 * turn lengths. Set DECAY_HALF_LIFE_SEC to Infinity for pure cumulative.
 *
 * Commands:
 *   /tps         notify the current model's tok/s
 *   /tps reset   reset the current model's counters
 *
 * --- Observation logging ---
 *
 * Every completed assistant turn (clean stop, not aborted/error) is written as
 * a JSON line to ~/.pi/agent/pi_observations.jsonl (or $PI_OBSERVATIONS_LOG).
 * This is the "raw facts" layer: model, prompt/gen tokens, decode + TTFT
 * latency, stop reason. No quant mapping or slug resolution happens here —
 * the model_id carries quant encoding (e.g. "Qwen3.6-27B-oQ8-mtp") and the
 * ingest script (pi-ingest skill) maps it to repo slugs + ALLOWED_QUANT.
 *
 * Fields per observation:
 *   model_id   – pi model name (provider-specific, quant-encoded)
 *   provider   – pi provider id ("omlx", "zai", "unsloth", etc.)
 *   prompt_tokens  – usage.input (context size)
 *   gen_tokens     – usage.output
 *   cache_read_tokens  – usage.cacheRead (prefix cache hit)
 *   cache_write_tokens – usage.cacheWrite (prefix cache miss fill)
 *   reasoning_tokens   – usage.reasoning (thinking tokens, subset of output)
 *   decode_seconds     – wall time from first delta to message_end
 *   ttft_seconds       – wall time from message_start to first delta
 *   stop_reason        – "stop" | "length" | "toolUse"
 *   timestamp          – ISO 8601 of message_end
 */

import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

type ModelKey = string;

interface Counters {
	tokens: number;
	seconds: number;
	chars: number; // accumulated output chars, for calibrating the live ratio
	lastSampleMs: number | undefined;
}

interface LiveState {
	streamStartMs: number | undefined; // set on first text/thinking delta (excludes TTFT)
	chars: number; // accumulated from text_delta + thinking_delta + toolcall_delta
	usageOutput: number; // max partial.usage.output seen (0 if provider doesn't stream it)
	lastRenderMs: number;
}

const RENDER_INTERVAL_MS = 200;
// Live estimate is suppressed until both thresholds pass; before that the
// decayed rest number stays visible (no noisy dip at turn start).
const MIN_STREAM_SEC = 0.3;
const MIN_EST_TOKENS = 20;
// Self-calibrating chars-per-token ratio. The live estimate divides accumulated
// chars by a ratio learned from completed turns on this model (decayed
// chars/tokens), blended with a prior until enough data accrues. 4 = prose
// default; a good prior here because reasoning/thinking output (natural
// language) dominates token count for thinking models, even when visible
// output is code.
const PRIOR_CHARS_PER_TOKEN = 4;
const PRIOR_WEIGHT_TOKENS = 500; // ~this much output before measured beats prior
// Half-life for the cumulative average, in seconds. Each new sample decays
// prior samples by 0.5^(Δt/HALF_LIFE). Short sessions are effectively
// cumulative; long sessions trend toward recent behavior (countering
// context-growth slowdown and provider-load drift). Set to Infinity to
// disable decay (pure cumulative).
const DECAY_HALF_LIFE_SEC = 30 * 60; // 30 min
// Physical plausibility cap for decode tps (tok/s). No current hardware
// decodes LLM tokens faster than this; observations that exceed it are
// measurement failures — specifically non-incremental "end-flush" responses
// where the provider buffers the whole output and emits it at completion, so
// the first-delta-to-message_end window (~0s) doesn't represent real decode
// time. Such observations are skipped (not logged, not counted in the running
// average) rather than emitting a misleading near-zero decode_seconds. Finer
// per-model ceiling filtering (memory_bandwidth / active_bytes_per_token)
// belongs in the ingest layer, which knows the hardware + model.
const IMPLAUSIBLE_DECODE_TPS = 500;

// --- Observation logger ---

function obsLogPath(): string {
	const env = process.env["PI_OBSERVATIONS_LOG"];
	if (env) return env;
	const piDir = join(homedir(), ".pi", "agent");
	return join(piDir, "pi_observations.jsonl");
}

/** Ensure the log file exists. */
function initObsLog(path: string): void {
	if (!existsSync(path)) {
		mkdirSync(join(path, ".."), { recursive: true });
		writeFileSync(path, "", "utf-8");
	}
}

function appendObs(path: string, obs: Record<string, unknown>): void {
	appendFileSync(path, JSON.stringify(obs) + "\n", "utf-8");
}

/** Produce a unique-ish id for dedup on re-ingest. */
function obsId(provider: string, modelId: string, ts: string, input: number, output: number): string {
	const shortTs = ts.replace(/[-:Z.]/g, "").slice(2, 20);
	return `obs-pi-${provider}-${modelId.replace(/[^a-zA-Z0-9_-]/g, "-")}-${shortTs}-i${input}-o${output}`;
}

function fmtTps(tokens: number, seconds: number): string {
	if (seconds <= 0) return "—";
	return (tokens / seconds).toFixed(1);
}

// Bayesian shrinkage: blend the measured chars/token from past turns with a
// prose prior, weighted by sample size.
function calibratedRatio(c: Counters | undefined): number {
	if (!c || c.tokens <= 0) return PRIOR_CHARS_PER_TOKEN;
	const w = c.tokens / (c.tokens + PRIOR_WEIGHT_TOKENS);
	const measured = c.chars / c.tokens;
	return w * measured + (1 - w) * PRIOR_CHARS_PER_TOKEN;
}

/** Resolve the prompt/context token count for an observation.
 *
 * Prefer the turn_start capture (most accurate: total system+history context
 * before the LLM call). For tool-result continuations — where the model is
 * called again within one user turn after a tool runs, so no fresh turn_start
 * fires — fall back to pi's context tracking read at message_end, then to
 * usage.input. Without the message_end fallback, continuations logged
 * prompt_tokens=0 whenever usage.input wasn't populated at message_end (it
 * resolves later in the persisted message). Returns null when nothing is
 * available (logged as prompt_tokens: null). */
function resolveContextTokens(turnCtx: number | undefined, ctxUsage: number | undefined, input: number): number | null {
	return turnCtx ?? ctxUsage ?? (input > 0 ? input : null);
}

/** True if a decode measurement is physically plausible (not an end-flush
 *  artifact). See IMPLAUSIBLE_DECODE_TPS. */
function isPlausibleDecode(output: number, elapsedSec: number): boolean {
	return elapsedSec > 0 && output / elapsedSec <= IMPLAUSIBLE_DECODE_TPS;
}

export default function (pi: ExtensionAPI) {
	// Per-model cumulative counters (authoritative, from usage.output).
	const stats = new Map<ModelKey, Counters>();
	let currentKey: ModelKey | undefined;

	// Observation log path (lazy init on first write).
	let obsPath: string | undefined;
	let obsInitialized = false;

	// Track message_start timestamp for TTFT.
	let messageStartMs: number | undefined;

	// Capture context size at turn_start (before LLM call), when it's most
	// accurate. Stores the total context (system + history), which is what
	// determines decode speed (KV cache size).
	let turnContextTokens: number | undefined;

	let live: LiveState | undefined;

	function keyFor(model: { provider: string; id: string } | undefined): ModelKey | undefined {
		return model ? `${model.provider}/${model.id}` : undefined;
	}

	function render(ctx: ExtensionContext, key: ModelKey | undefined) {
		const theme = ctx.ui.theme;
		const c = key ? stats.get(key) : undefined;
		const tps = c ? fmtTps(c.tokens, c.seconds) : "—";
		ctx.ui.setStatus("tps", theme.fg("dim", `${tps} tok/s  │`));
	}

	function renderLive(ctx: ExtensionContext) {
		if (!live || live.streamStartMs == null) return;
		const now = Date.now();
		if (now - live.lastRenderMs < RENDER_INTERVAL_MS) return;

		const elapsed = (now - live.streamStartMs) / 1000;
		const ratio = calibratedRatio(currentKey ? stats.get(currentKey) : undefined);
		const estimate = live.usageOutput > 0 ? live.usageOutput : live.chars / ratio;
		if (elapsed < MIN_STREAM_SEC || estimate < MIN_EST_TOKENS) return;
		live.lastRenderMs = now;

		const tps = (estimate / elapsed).toFixed(1);
		const prefix = live.usageOutput > 0 ? "" : "≈";
		ctx.ui.setStatus("tps", ctx.ui.theme.fg("dim", `${prefix}${tps} tok/s  │`));
	}

	pi.on("session_start", (_event, ctx) => {
		currentKey = keyFor(ctx.model);
		render(ctx, currentKey);
	});

	pi.on("model_select", (event, ctx) => {
		currentKey = keyFor(event.model);
		render(ctx, currentKey);
	});

	pi.on("turn_start", (_event, ctx) => {
		const ctxUsage = ctx.getContextUsage();
		turnContextTokens = ctxUsage?.tokens ?? undefined;
	});

	pi.on("message_start", (event, _ctx) => {
		if (event.message.role === "assistant") {
			messageStartMs = Date.now();
			live = { streamStartMs: undefined, chars: 0, usageOutput: 0, lastRenderMs: 0 };
		}
	});

	pi.on("message_update", (event, ctx) => {
		if (!live) return;
		const ev = event.assistantMessageEvent;

		if (ev.type === "text_delta" || ev.type === "thinking_delta" || ev.type === "toolcall_delta") {
			if (live.streamStartMs == null) live.streamStartMs = Date.now();
			live.chars += ev.delta.length;
		}

		const partialOutput = (event.message as { usage?: { output?: number } }).usage?.output ?? 0;
		if (partialOutput > live.usageOutput) live.usageOutput = partialOutput;

		renderLive(ctx);
	});

	pi.on("message_end", (event, ctx) => {
		if (event.message.role !== "assistant") return;
		const wasLive = live;
		live = undefined;
		if (wasLive == null) return;

		const nowMs = Date.now();
		const elapsed = wasLive.streamStartMs != null ? (nowMs - wasLive.streamStartMs) / 1000 : 0;
		const msg = event.message as {
			usage?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number; reasoning?: number };
			stopReason?: string;
		};

		// --- Observation logging ---
		const output = msg.usage?.output ?? 0;
		const input = msg.usage?.input ?? 0;
		const clean = msg.stopReason == null || msg.stopReason === "stop" || msg.stopReason === "length" || msg.stopReason === "toolUse";
		// Skip non-incremental "end-flush" responses (decode measured over an
		// implausibly short window) — see IMPLAUSIBLE_DECODE_TPS. Gates both the
		// observation log and the running average.
		const loggable = clean && output > 0 && isPlausibleDecode(output, elapsed);

		if (loggable) {
			const model = event.message as { model?: string; provider?: string };
			const modelId = model.model ?? "unknown";
			const provider = model.provider ?? "unknown";
			const ts = new Date(nowMs).toISOString();

			const ttft = messageStartMs != null && wasLive.streamStartMs != null
				? (wasLive.streamStartMs - messageStartMs) / 1000
				: undefined;

			// Context size: prefer turn_start capture (total context = system +
			// history, determines KV cache size). For tool-result continuations
			// (no fresh turn_start), fall back to pi's context tracking at
			// message_end, then usage.input. Without the message_end fallback,
			// continuations logged prompt_tokens=0 when usage.input wasn't
			// populated at message_end.
			const contextTokens = resolveContextTokens(turnContextTokens, ctx.getContextUsage()?.tokens, input);
			turnContextTokens = undefined; // reset for next turn

			const obs: Record<string, unknown> = {
				id: obsId(provider, modelId, ts, contextTokens ?? 0, output),
				model_id: modelId,
				provider: provider,
				prompt_tokens: contextTokens,
				gen_tokens: output,
				cache_read_tokens: msg.usage?.cacheRead ?? 0,
				cache_write_tokens: msg.usage?.cacheWrite ?? 0,
				reasoning_tokens: msg.usage?.reasoning ?? 0,
				decode_seconds: Number(elapsed.toFixed(4)),
				ttft_seconds: ttft !== undefined ? Number(ttft.toFixed(4)) : null,
				stop_reason: msg.stopReason ?? "unknown",
				timestamp: ts,
			};

			// Lazy init the log file path and write.
			if (!obsInitialized) {
				obsPath = obsLogPath();
				initObsLog(obsPath);
				obsInitialized = true;
			}
			appendObs(obsPath!, obs);
		}

		// --- Decayed average (existing behavior) ---
		if (loggable) {
			const key = currentKey ?? "unknown";
			const c = stats.get(key) ?? { tokens: 0, seconds: 0, chars: 0, lastSampleMs: undefined };
			const now = Date.now();
			if (c.lastSampleMs != null && DECAY_HALF_LIFE_SEC < Infinity) {
				const dt = (now - c.lastSampleMs) / 1000;
				const decay = Math.pow(0.5, dt / DECAY_HALF_LIFE_SEC);
				c.tokens *= decay;
				c.seconds *= decay;
				c.chars *= decay;
			}
			c.tokens += output;
			c.seconds += elapsed;
			c.chars += wasLive.chars;
			c.lastSampleMs = now;
			stats.set(key, c);
		}

		render(ctx, currentKey);
	});

	pi.registerCommand("tps", {
		description: "Show or reset tokens-per-second for the current model",
		handler: async (args, ctx) => {
			const key = currentKey ?? keyFor(ctx.model);
			if (args.trim() === "reset") {
				if (key) stats.delete(key);
				render(ctx, key);
				ctx.ui.notify(`Reset tok/s for ${key ?? "current model"}`, "info");
				return;
			}
			const c = key ? stats.get(key) : undefined;
			const tps = c ? fmtTps(c.tokens, c.seconds) : "—";
			const n = c ? Math.round(c.seconds) : 0;
			const ratio = calibratedRatio(c);
			const mode = live ? (live.usageOutput > 0 ? "live (exact)" : "live (estimated)") : "at rest";
			ctx.ui.notify(`${key ?? "current model"}: ${tps} tok/s · ${ratio.toFixed(2)}c/t (${n}s sampled, ${mode})`, "info");
		},
	});
}
