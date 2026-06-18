// Per-voice visual identity. The hues are defined in globals.css; here we map each
// persona key to its hue, a short broadcast-style tag, and a fallback label so the
// commentary feed makes the four voices recognizable without a rainbow.

export interface VoiceMeta {
  label: string;
  tag: string;
  hue: string; // a CSS custom property reference
}

export const VOICE_META: Record<string, VoiceMeta> = {
  broadcast: { label: "Broadcast Box", tag: "TV", hue: "var(--voice-broadcast)" },
  radio: { label: "Radio Call", tag: "RADIO", hue: "var(--voice-radio)" },
  analyst: { label: "The Tactician", tag: "ANALYST", hue: "var(--voice-analyst)" },
  text: { label: "The Wire", tag: "WIRE", hue: "var(--voice-text)" },
};

export function voiceHue(key: string): string {
  return VOICE_META[key]?.hue ?? "var(--accent)";
}

export function voiceTag(key: string): string {
  return VOICE_META[key]?.tag ?? key.toUpperCase();
}
