/**
 * Tests for the observation logging logic in the tps extension.
 *
 * These test the pure functions (obsId, obsLogPath, observation format)
 * without needing pi running. The pi event handlers (message_start/end)
 * are integration-level and tested by running the extension in pi.
 */
import { appendFileSync, existsSync, mkdirSync, rmSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { tmpdir } from "node:os";

// --- Test helpers ---

function randomId(): string {
	return Math.random().toString(36).slice(2, 10);
}

function assert(condition: boolean, msg: string): void {
	if (!condition) throw new Error(`FAIL: ${msg}`);
	console.log(`  ✓ ${msg}`);
}

function assertEq<T>(actual: T, expected: T, msg: string): void {
	if (actual !== expected) throw new Error(`FAIL: ${msg}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`);
	console.log(`  ✓ ${msg}`);
}

function assertMatch(actual: string, pattern: RegExp, msg: string): void {
	if (!pattern.test(actual)) throw new Error(`FAIL: ${msg}\n  string: ${JSON.stringify(actual)}\n  pattern: ${pattern}`);
	console.log(`  ✓ ${msg}`);
}

// --- Functions under test (copied from index.ts) ---

function obsLogPath(envOverride?: string): string {
	if (envOverride) return envOverride;
	const piDir = join(homedir(), ".pi", "agent");
	return join(piDir, "pi_observations.jsonl");
}

function obsId(provider: string, modelId: string, ts: string, input: number, output: number): string {
	const shortTs = ts.replace(/[-:Z.]/g, "").slice(2, 20);
	return `obs-pi-${provider}-${modelId.replace(/[^a-zA-Z0-9_-]/g, "-")}-${shortTs}-i${input}-o${output}`;
}

// --- Tests ---

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void) {
	try {
		fn();
		passed++;
		console.log(`\n${name}`);
	} catch (e) {
		failed++;
		console.error(`\n✖ ${name}`);
		console.error(`  ${(e as Error).message}`);
	}
}

// --- obsLogPath ---

test("obsLogPath default", () => {
	const path = obsLogPath();
	assertMatch(path, /\/\.pi\/agent\/pi_observations\.jsonl$/, "default path ends in .pi/agent/pi_observations.jsonl");
});

test("obsLogPath with env override", () => {
	const path = obsLogPath("/tmp/test-obs.jsonl");
	assert(path === "/tmp/test-obs.jsonl", "env override respected");
});

// --- obsId ---

test("obsId format", () => {
	const ts = "2026-07-26T12:00:00.000Z";
	const id = obsId("omlx", "Qwen3.6-27B-oQ8-mtp", ts, 12345, 678);
	assertMatch(id, /^obs-pi-omlx-Qwen3-6-27B-oQ8-mtp-/, "prefix with provider + sanitized modelId");
	assertMatch(id, /-i12345-o678$/, "suffix with input/output counts");
	assert(id.length < 200, "id is not excessively long");
});

test("obsId handles special chars in modelId", () => {
	const ts = "2026-07-26T12:00:00.000Z";
	const id = obsId("unsloth", "unsloth/DeepSeek-V4-Flash-GGUF", ts, 5000, 200);
	// Slashes and special chars should be replaced
	assert(!id.includes("/"), "no slashes in id");
	assert(!id.includes("."), "no dots in id");
	assertMatch(id, /-unsloth-unsloth-DeepSeek-V4-Flash-GGUF-/, "modelId sanitized");
});

test("obsId uniqueness", () => {
	const ts1 = "2026-07-26T12:00:00.000Z";
	const ts2 = "2026-07-26T12:00:01.000Z";
	const id1 = obsId("omlx", "A", ts1, 100, 50);
	const id2 = obsId("omlx", "A", ts2, 100, 50);
	assert(id1 !== id2, "different timestamps produce different ids");
	const id3 = obsId("omlx", "A", ts1, 200, 50);
	assert(id1 !== id3, "different input counts produce different ids");
});

// --- Observation format ---

test("observation JSON shape", () => {
	const obs = {
		id: "obs-pi-omlx-Qwen-12345-678",
		model_id: "Qwen3.6-27B-oQ8-mtp",
		provider: "omlx",
		prompt_tokens: 12345,
		gen_tokens: 678,
		cache_read_tokens: 10000,
		cache_write_tokens: 2345,
		reasoning_tokens: 200,
		decode_seconds: 12.34,
		ttft_seconds: 0.56,
		stop_reason: "stop",
		timestamp: "2026-07-26T12:00:00.000Z",
	};
	const json = JSON.stringify(obs);
	const parsed = JSON.parse(json);
	assertEq(parsed.model_id, "Qwen3.6-27B-oQ8-mtp", "model_id survives round-trip");
	assertEq(parsed.prompt_tokens, 12345, "prompt_tokens survives round-trip");
	assertEq(parsed.decode_seconds, 12.34, "decode_seconds survives round-trip");
	assertEq(parsed.ttft_seconds, 0.56, "ttft_seconds survives round-trip");
	assert(parsed.stop_reason === "stop", "stop_reason survives round-trip");
});

test("observation with null TTFT", () => {
	const obs = {
		id: "obs-pi-test",
		model_id: "test-model",
		provider: "test",
		prompt_tokens: 100,
		gen_tokens: 50,
		cache_read_tokens: 0,
		cache_write_tokens: 0,
		reasoning_tokens: 0,
		decode_seconds: 5.0,
		ttft_seconds: null,
		stop_reason: "toolUse",
		timestamp: "2026-07-26T12:00:00.000Z",
	};
	const json = JSON.stringify(obs);
	const parsed = JSON.parse(json);
	assert(parsed.ttft_seconds === null, "null TTFT preserved");
});

// --- Log file append ---

test("append and read back", () => {
	const tmpFile = join(tmpdir(), `tps-test-${randomId()}.jsonl`);
	try {
		// Simulate the extension's appendObs
		const obs1 = { id: "obs-1", model_id: "m1", prompt_tokens: 100, gen_tokens: 50 };
		const obs2 = { id: "obs-2", model_id: "m2", prompt_tokens: 200, gen_tokens: 100 };

		// Append using the imported appendFileSync
		appendFileSync(tmpFile, JSON.stringify(obs1) + "\n", "utf-8");
		appendFileSync(tmpFile, JSON.stringify(obs2) + "\n", "utf-8");

		// Read back
		const lines = readFileSync(tmpFile, "utf-8").trim().split("\n");
		assert(lines.length === 2, "two lines written");
		const parsed1 = JSON.parse(lines[0]);
		assertEq(parsed1.id, "obs-1", "first obs id correct");
		const parsed2 = JSON.parse(lines[1]);
		assertEq(parsed2.id, "obs-2", "second obs id correct");
	} finally {
		try { rmSync(tmpFile); } catch {}
	}
});

// --- Summary ---

console.log(`\n--- ${passed} passed, ${failed} failed ---`);
if (failed > 0) {
	console.error(`\n✖ ${failed} test(s) FAILED`);
	process.exit(1);
}
