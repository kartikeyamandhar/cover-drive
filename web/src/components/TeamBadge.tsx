import type { CSSProperties } from "react";
import { teamCode, teamColor } from "@/lib/teams";
import styles from "./TeamBadge.module.css";

// A colored monogram crest standing in for a team logo (legal to ship; reads as a badge).
export function TeamBadge({ team, size = 40 }: { team: string; size?: number }) {
  return (
    <span
      className={styles.badge}
      style={{ "--team": teamColor(team), "--size": `${size}px` } as CSSProperties}
      title={team}
      aria-label={team}
    >
      {teamCode(team)}
    </span>
  );
}
