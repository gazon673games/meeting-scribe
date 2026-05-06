export function applyAsrMetrics(state, event) {
  if (!state?.session) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      asrMetrics: {
        segDroppedTotal: Number(event.seg_dropped_total ?? state.session.asrMetrics?.segDroppedTotal ?? 0),
        segSkippedTotal: Number(event.seg_skipped_total ?? state.session.asrMetrics?.segSkippedTotal ?? 0),
        avgLatencyS: Number(event.avg_latency_s ?? state.session.asrMetrics?.avgLatencyS ?? 0),
        p95LatencyS: Number(event.p95_latency_s ?? state.session.asrMetrics?.p95LatencyS ?? 0),
        lagS: Number(event.lag_s ?? state.session.asrMetrics?.lagS ?? 0)
      }
    }
  };
}

export function upsertSessionSource(state, source) {
  if (!state?.session || !source?.name) {
    return state;
  }
  const sources = state.session.sources || [];
  const index = sources.findIndex((item) => item.name === source.name);
  const nextSources =
    index >= 0
      ? sources.map((item, itemIndex) => (itemIndex === index ? { ...item, ...source } : item))
      : [...sources, source];
  return {
    ...state,
    session: {
      ...state.session,
      sources: nextSources
    }
  };
}

export function removeSessionSource(state, source) {
  if (!state?.session) {
    return state;
  }
  const sourceName = String(source?.name || source || "");
  if (!sourceName) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      sources: (state.session.sources || []).filter((item) => item.name !== sourceName)
    }
  };
}
