import { Activity } from "lucide-react";

export function LatencyMeta({ session }) {
  const text = session.asrRunning ? "live" : "standby";
  return (
    <span className="column-meta">
      <Activity size={13} />
      {text}
    </span>
  );
}
