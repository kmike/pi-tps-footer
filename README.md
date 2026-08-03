# pi-tps-footer

[![test](https://github.com/kmike/pi-tps-footer/actions/workflows/test.yml/badge.svg)](https://github.com/kmike/pi-tps-footer/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/github/license/kmike/pi-tps-footer?color=blue)](https://github.com/kmike/pi-tps-footer/blob/main/LICENSE)
[![pi extension](https://img.shields.io/badge/pi-extension-8A4FFF)](https://github.com/earendil-works/pi-coding-agent)

Shows tokens-per-second in [pi](https://github.com/earendil-works/pi-coding-agent)'s
footer: a live estimate while the model is streaming, and a running average
when idle. Counts are kept per model, so switching models never mixes their
numbers.

Optionally, it can also log every completed turn (model, token counts, decode
and TTFT latency, stop reason) to a JSONL file for offline analysis — see
below.

## Install

```bash
pi install git:github.com/kmike/pi-tps-footer
```

## Commands

| Command | Effect |
| --- | --- |
| `/tps` | Notify the current model's tok/s |
| `/tps reset` | Reset the current model's counters |
| `/tps log` | Show observation-log state and path |
| `/tps log on` / `/tps log off` | Enable/disable the observation log (persisted) |

## Observation log

Off by default. When enabled with `/tps log on`, each completed turn is
appended as a JSON line to `~/.pi/agent/pi-tps-footer/observations.jsonl`
(override with the `PI_OBSERVATIONS_LOG` environment variable).

## Development

```bash
npm test
```

## License

MIT
