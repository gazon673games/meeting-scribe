import React from "react";
import { Cpu } from "lucide-react";

import { normalizeAssistantProfiles } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { AssistantProfileDetails } from "./assistant/AssistantProfileDetails";
import { AssistantProfilesSidebar } from "./assistant/AssistantProfilesSidebar";
import { newProfile, uniqueProfileId } from "./assistant/profileUtils";

export function AssistantSettings({ draft, onChange }) {
  const assistantEnabled = Boolean(draft.assistantEnabled);
  const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
  const selectedProfileId = draft.assistantSelectedProfileId || profiles[0]?.id || "";
  const selectedIndex = Math.max(0, profiles.findIndex((profile) => profile.id === selectedProfileId));
  const selectedProfile = profiles[selectedIndex] || profiles[0];

  const [pingStates, setPingStates] = React.useState({});
  const [llmModels, setLlmModels] = React.useState([]);
  const [llmLoaded, setLlmLoaded] = React.useState(false);

  const hasLocalProfile = profiles.some((profile) => (profile.provider || "codex") !== "codex");
  React.useEffect(() => {
    if (!hasLocalProfile || llmLoaded) return;
    setLlmLoaded(true);
    meetingScribeClient
      .request("list_llm_models", {})
      .then((result) => setLlmModels(result?.models || []))
      .catch(() => {});
  }, [hasLocalProfile, llmLoaded]);

  const updateProfile = React.useCallback((index, patch) => {
    const next = profiles.map((profile, profileIndex) => (profileIndex === index ? { ...profile, ...patch } : profile));
    onChange({ assistantProfiles: next });
  }, [onChange, profiles]);

  const updateSelected = React.useCallback((patch) => {
    if (!selectedProfile) return;
    updateProfile(selectedIndex, patch);
  }, [selectedIndex, selectedProfile, updateProfile]);

  const addProfile = React.useCallback((provider = "codex") => {
    const id = uniqueProfileId(profiles, provider);
    onChange({
      assistantProfiles: [...profiles, newProfile(id, provider)],
      assistantSelectedProfileId: id
    });
  }, [onChange, profiles]);

  const removeProfile = React.useCallback((index) => {
    const removed = profiles[index];
    const next = profiles.filter((_, profileIndex) => profileIndex !== index);
    const fallback = next[Math.max(0, Math.min(index, next.length - 1))]?.id || "";
    onChange({
      assistantProfiles: next,
      assistantSelectedProfileId:
        draft.assistantSelectedProfileId === removed.id ? fallback : draft.assistantSelectedProfileId
    });
  }, [draft.assistantSelectedProfileId, onChange, profiles]);

  const pingProfile = React.useCallback(async (profile) => {
    setPingStates((state) => ({ ...state, [profile.id]: { busy: true } }));
    try {
      const result = await meetingScribeClient.request("ping_assistant_provider", {
        providerId: profile.provider || "codex",
        profileId: profile.id
      });
      setPingStates((state) => ({ ...state, [profile.id]: { busy: false, ...result } }));
    } catch (error) {
      setPingStates((state) => ({
        ...state,
        [profile.id]: { busy: false, ok: false, errorCode: "ping_failed", message: String(error?.message || error) }
      }));
    }
  }, []);

  const cachedLocalModels = llmModels
    .filter((model) => model.cached)
    .map((model) => String(model.modelAlias || model.name || ""))
    .filter(Boolean);

  const cachedGgufModels = llmModels
    .filter((model) => model.cached && model.path)
    .map((model) => ({ label: String(model.label || model.name || model.path), path: String(model.path) }));

  return (
    <CollapsibleSection title="Assistant" defaultOpen={false}>
      <div className="assistant-settings-shell">
        <button
          aria-pressed={assistantEnabled}
          className={`feature-toggle proxy-toggle ${assistantEnabled ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ assistantEnabled: !assistantEnabled })}
        >
          <Cpu size={15} />
          <span>Enable Assistant</span>
          <b />
        </button>

        <div className="assistant-profile-workbench">
          <AssistantProfilesSidebar
            profiles={profiles}
            selectedProfileId={selectedProfile?.id}
            onSelect={(id) => onChange({ assistantSelectedProfileId: id })}
            onDelete={removeProfile}
            onAdd={addProfile}
          />

          {selectedProfile ? (
            <AssistantProfileDetails
              assistantEnabled={assistantEnabled}
              cachedGgufModels={cachedGgufModels}
              cachedLocalModels={cachedLocalModels}
              ping={pingStates[selectedProfile.id]}
              profile={selectedProfile}
              onDelete={() => removeProfile(selectedIndex)}
              onPing={() => pingProfile(selectedProfile)}
              onUpdate={updateSelected}
            />
          ) : (
            <div className="assistant-profile-detail assistant-profile-empty">
              <p>No profiles configured.</p>
              <p>Add a profile using the buttons on the left.</p>
            </div>
          )}
        </div>
      </div>
    </CollapsibleSection>
  );
}
