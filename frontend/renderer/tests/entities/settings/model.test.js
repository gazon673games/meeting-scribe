import { expect, test } from 'vitest';

import {
  applySettingsToConfig,
  makeSettingsDraft,
} from '../../../src/entities/settings/model';

test('preserves Balanced profile and every model subdirectory across a settings round trip', () => {
  const config = {
    ui: { profile: 'Balanced' },
    models: {
      cache_dir: 'D:/cache',
      asr_subdir: 'speech',
      diarization_subdir: 'speakers',
      llm_subdir: 'chat',
    },
  };
  const result = applySettingsToConfig(config, makeSettingsDraft(config));
  expect(result.models).toMatchObject(config.models);
  expect(result.ui.profile).toBe('Balanced');
});

test('normalizes duplicate assistant ids to unique stable selections', () => {
  const draft = makeSettingsDraft({
    codex: { profiles: [{ id: 'p' }, { id: 'p' }, { id: 'p-2' }] },
  });
  const ids = draft.assistantProfiles.map((profile) => profile.id);
  expect(new Set(ids).size).toBe(3);
  expect(
    makeSettingsDraft(applySettingsToConfig({}, draft)).assistantProfiles.map(
      (profile) => profile.id,
    ),
  ).toEqual(ids);
});
