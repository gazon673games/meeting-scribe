import React from "react";
import { Plus, Trash2 } from "lucide-react";

import { normalizeProvider, modelBasename, runtimeLabel } from "./profileUtils";

export function AssistantProfilesSidebar({ profiles, selectedProfileId, onSelect, onDelete, onAdd }) {
  return (
    <div className="assistant-profile-sidebar">
      <div className="assistant-profile-sidebar-head">
        <span>Profiles</span>
        <span>{profiles.length}</span>
      </div>

      <div className="assistant-profile-list">
        {profiles.map((profile, index) => {
          const isCodex = normalizeProvider(profile.provider) === "codex";
          const selected = profile.id === selectedProfileId;
          return (
            <div className={`assistant-profile-list-item ${selected ? "selected" : ""}`} key={profile.id || index}>
              <button className="assistant-profile-list-select" type="button" onClick={() => onSelect(profile.id)}>
                <span className="assistant-profile-list-main">
                  <b>{profile.label || profile.id}</b>
                  <small>
                    {isCodex ? "Codex CLI" : runtimeLabel(profile.provider)}
                    {!isCodex && profile.model ? ` · ${modelBasename(profile.model)}` : ""}
                  </small>
                </span>
                <span className={`profile-mode-badge ${isCodex ? "online" : "offline"}`}>
                  {isCodex ? "online" : "local"}
                </span>
              </button>
              <button className="model-row-btn danger" title="Delete" type="button" onClick={() => onDelete(index)}>
                <Trash2 size={12} />
              </button>
            </div>
          );
        })}
      </div>

      <div className="assistant-add-row">
        <button className="model-download-button" type="button" onClick={() => onAdd("codex")}>
          <Plus size={13} />
          Codex
        </button>
        <button className="model-download-button" type="button" onClick={() => onAdd("local")}>
          <Plus size={13} />
          GGUF
        </button>
        <button className="model-download-button" type="button" onClick={() => onAdd("ollama")}>
          <Plus size={13} />
          Ollama
        </button>
      </div>
    </div>
  );
}
