import type { CSSProperties } from "react";
import type { PersonaInfo } from "@/lib/types";
import { voiceHue, voiceTag } from "@/lib/personas";
import styles from "./PersonaSwitcher.module.css";

interface Props {
  personas: PersonaInfo[];
  active: string;
  onSelect: (key: string) => void;
}

// The headline control: switch the voice and the same passage re-renders. The model owns
// style, the facts stay fixed, so this is the visible proof the fine-tune controls voice.
export function PersonaSwitcher({ personas, active, onSelect }: Props) {
  return (
    <div className={styles.wrap}>
      <span className={styles.label}>Voice</span>
      <div className={styles.group} role="group" aria-label="Commentary voice">
        {personas.map((p) => {
          const isActive = p.key === active;
          return (
            <button
              key={p.key}
              type="button"
              className={`${styles.chip} ${isActive ? styles.active : ""}`}
              style={{ "--voice": voiceHue(p.key) } as CSSProperties}
              aria-pressed={isActive}
              onClick={() => onSelect(p.key)}
              title={p.instruction}
            >
              <span className={styles.tag}>{voiceTag(p.key)}</span>
              <span className={styles.name}>{p.display_name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
