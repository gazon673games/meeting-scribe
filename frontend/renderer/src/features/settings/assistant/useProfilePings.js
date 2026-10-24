import React from 'react';

import { meetingScribeClient } from '../../../shared/api/meetingScribeClient';

const pingState = (payload, busy, fingerprint) => ({
  busy,
  ok: payload.ok ?? false,
  errorCode: payload.errorCode ?? '',
  message: payload.message ?? '',
  latencyMs: payload.latencyMs ?? null,
  ts: payload.ts ?? null,
  _fingerprint: fingerprint,
});

export const useProfilePings = (profiles) => {
  const [pingStates, setPingStates] = React.useState(() => new Map());
  const scope = React.useRef({ active: false, pending: new Map() });

  React.useLayoutEffect(() => {
    const fingerprints = new Map(
      profiles.map((profile) => [profile.id, JSON.stringify(profile)]),
    );
    for (const [id, ticket] of scope.current.pending) {
      if (fingerprints.get(id) !== ticket.fingerprint) {
        scope.current.pending.delete(id);
      }
    }
    setPingStates((current) => {
      const next = new Map();
      for (const [id, state] of current) {
        if (fingerprints.get(id) === state._fingerprint) next.set(id, state);
      }
      return next.size === current.size ? current : next;
    });
  }, [profiles]);

  const consume = React.useCallback((owner, profileId, requestId, payload) => {
    const ticket = owner.pending.get(profileId);
    if (!owner.active || !ticket || ticket.id !== requestId) return;
    owner.pending.delete(profileId);
    setPingStates((current) =>
      new Map(current).set(
        profileId,
        pingState(payload, false, ticket.fingerprint),
      ),
    );
  }, []);

  React.useEffect(() => {
    const owner = { active: true, pending: new Map() };
    scope.current = owner;
    const unsubscribe = meetingScribeClient.onBackendEvent?.((event) => {
      if (event?.type === 'assistant_ping_result') {
        consume(owner, event.profileId, event.requestId, event);
      }
    });
    return () => {
      owner.active = false;
      owner.pending.clear();
      unsubscribe?.();
    };
  }, [consume]);

  const pingProfile = React.useCallback(
    async (profile) => {
      const owner = scope.current;
      if (!owner.active) return;
      const ticket = {
        id: globalThis.crypto.randomUUID(),
        fingerprint: JSON.stringify(profile),
      };
      owner.pending.set(profile.id, ticket);
      setPingStates((current) =>
        new Map(current).set(
          profile.id,
          pingState({}, true, ticket.fingerprint),
        ),
      );
      try {
        const result = await meetingScribeClient.request(
          'ping_assistant_provider',
          {
            providerId: profile.provider || 'codex',
            profileId: profile.id,
            profile,
            requestId: ticket.id,
          },
        );
        if (!result?.pending) consume(owner, profile.id, ticket.id, result);
      } catch (error) {
        consume(owner, profile.id, ticket.id, {
          ok: false,
          errorCode: 'ping_failed',
          message: String(error?.message || error),
        });
      }
    },
    [consume],
  );

  return { pingStates, pingProfile };
};
