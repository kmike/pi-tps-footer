# pi-tps-footer

Tokens-per-second in [pi](https://github.com/earendil-works/pi-coding-agent)'s footer, plus an opt-in per-turn observation log.

## What it shows

Two views over the same footer slot:

- **While streaming:** live instantaneous tok/s, updated ~5×/sec.
- **At rest:** decayed average tok/s for the current model, updated at each
  completed turn. Prior samples fade with a 30-min half-life, so long sessions
  track recent behavior; short sessions behave like a plain cumulative average.

Counters are bucketed **per model**, so switching models never blends
throughput from different models; cycling back to a model resumes its own
average.

Live token counting is provider-dependent:

- Anthropic streams cumulative `output_tokens` live → exact number.
- OpenAI-compatible providers emit usage only in the final chunk → live number
  is a char-based estimate, shown with a `≈`. The chars-per-token ratio is
  *calibrated* from completed turns on the same model (Bayesian shrinkage
  toward a prose-leaning prior), so it converges to the model's true ratio for
  your content mix after a few turns.

Timing starts at the first streamed delta (not `message_start`), so both
numbers measure decode throughput and exclude TTFT. End-flush responses
(providers that buffer the whole output and emit it at completion, implying
physically impossible decode speeds) are detected and skipped rather than
polluting the average.

## Install

```bash
pi install git:github.com/kmike/pi-tps-footer
```

## Commands

| Command | Effect |
| --- | --- |
| `/tps` | Notify the current model's tok/s (with calibration ratio and sample size) |
| `/tps reset` | Reset the current model's counters |
| `/tps log` | Show observation-log state and path |
| `/tps log on` / `/tps log off` | Enable/disable the observation log (persisted) |

## Observation log (opt-in)

Off by default. When enabled with `/tps log on`, every completed assistant
turn (clean stop, not aborted/error) is appended as a JSON line to:

```
~/.pi/agent/pi-tps-footer/observations.jsonl
```

Override the location with the `PI_OBSERVATIONS_LOG` environment variable.
The on/off state persists in `~/.pi/agent/pi-tps-footer/state.json`.

Fields per observation: `id` (dedup key), `model_id`, `provider`,
`prompt_tokens` (total context: system + history), `gen_tokens`,
`cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`,
`decode_seconds` (first delta → end), `ttft_seconds`, `stop_reason`,
`timestamp`.

This is a "raw facts" layer meant for offline analysis — e.g. comparing
decode throughput across models, quants, and providers.

## Development

```bash
node tests/obs-log.test.ts   # or: npm test
```

## License

MIT
