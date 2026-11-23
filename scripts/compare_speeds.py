#!/usr/bin/env python3
"""Run alternating Gazebo speed trials from a Docker-enabled host."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_STATES = {
    'corner_recovery', 'conflict_recovery', 'path_end_recovery', 'short_exit_recovery',
}


def summarize(folder):
    capture = next((path for path in folder.iterdir() if path.name.isdecimal()), None)
    if capture is None:
        return {'status': 'no_capture'}
    trace = capture/'control.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    stop = capture/'terminal_stop.json'
    terminal = json.loads(stop.read_text()) if stop.exists() else {}
    start = next((row['time'] for row in rows if row['speed'] > 0), None)
    states = [row['state'] for row in rows]
    camera = next(iter(sorted(capture.glob('camera_*.json'))), None)
    snapshot = json.loads(camera.read_text()) if camera is not None else {}
    return {
        'status': 'finished' if terminal.get('stop_reason') == 'terminal_t' else 'incomplete',
        'capture': capture.name,
        'revision': terminal.get('controller_revision', snapshot.get('controller_revision')),
        'elapsed': round(terminal['time']-start, 3) if terminal and start is not None else None,
        'max_odom_speed': max((row['odom_speed'] or 0 for row in rows), default=None),
        'recoveries': sum(state in RECOVERY_STATES and previous not in RECOVERY_STATES
                          for previous, state in zip([None]+states, states)),
        'conflict_stops': sum(state == 'visual_conflict_stop' and previous != state
                              for previous, state in zip([None]+states, states)),
    }


def stop_runner(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    process.wait(timeout=20)


def run_trial(folder, speed, timeout, headless):
    folder.mkdir()
    env = dict(os.environ, PROJECT1_STRAIGHT_SPEED=speed, PROJECT1_CAPTURE_DIR=str(folder))
    command = ['bash', 'scripts/run_docker.sh']
    if headless:
        command.append('--headless')
    deadline = time.monotonic()+timeout
    outcome = 'runner_exit'
    with (folder/'runner.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if any(folder.glob('[0-9]*/terminal_stop.json')):
                    outcome = 'terminal_recorded'
                    break
                if time.monotonic() >= deadline:
                    outcome = 'timeout'
                    break
                time.sleep(.5)
        finally:
            stop_runner(process)
    return dict(summarize(folder), speed=speed, outcome=outcome, trial=folder.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=300, help='wall-clock seconds per trial')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.pairs < 1 or args.timeout <= 0:
        parser.error('--pairs and --timeout must be positive')
    if subprocess.run(['docker', 'info'], stdout=subprocess.DEVNULL).returncode:
        parser.exit(1, 'Run this script from a terminal with Docker access.\n')
    folder = ROOT/'captures'/f'speed_comparison_{time.time_ns()}'
    folder.mkdir(parents=True)
    print(folder, flush=True)
    results = defaultdict(list)
    for pair in range(1, args.pairs+1):
        speeds = ('0.23', '0.25') if pair % 2 else ('0.25', '0.23')
        for speed in speeds:
            print(f'Pair {pair}/{args.pairs} · {speed} m/s', flush=True)
            result = run_trial(folder/f'{pair:02d}_{speed}', speed, args.timeout, args.headless)
            results[speed].append(result)
            with (folder/'results.jsonl').open('a') as output:
                output.write(json.dumps(result)+'\n')
            print(result, flush=True)
            if result['status'] == 'no_capture' and result['outcome'] == 'runner_exit':
                parser.exit(1, f'No camera capture. See {folder/result["trial"]/"runner.log"}\n')
    for speed, trials in sorted(results.items()):
        times = [row['elapsed'] for row in trials if row['elapsed'] is not None]
        if times:
            print(f'{speed} m/s: {len(times)}/{len(trials)} finished, '
                  f'median {statistics.median(times):.2f} s, slowest {max(times):.2f} s')
        else:
            print(f'{speed} m/s: 0/{len(trials)} finished')
    revisions = {row['revision'] for trials in results.values() for row in trials if row.get('revision')}
    if len(revisions) != 1:
        print(f'Controller revisions recorded: {sorted(revisions)}')


if __name__ == '__main__':
    main()
