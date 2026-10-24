import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';

const script = fileURLToPath(
  new URL('../../test_complexity.js', import.meta.url),
);

test.each(['NaN', 'Infinity', '-1'])(
  'rejects an invalid complexity threshold: %s',
  (threshold) => {
    const result = spawnSync(
      process.execPath,
      [script, '--max-pct', threshold],
      { encoding: 'utf8' },
    );
    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/finite numbers|Invalid complexity limits/);
  },
);
