"use client";

import { useState, type CSSProperties } from "react";
import Image from "next/image";
import { TEAM_LOGOS_ENABLED, teamCode, teamColor } from "@/lib/teams";
import styles from "./TeamBadge.module.css";

// A colored monogram crest standing in for a team logo (legal to ship). If real artwork is
// provided (NEXT_PUBLIC_TEAM_LOGOS=1 and web/public/teams/<CODE>.png), it renders the image
// and falls back to the crest if the file is missing. We never bundle trademarked logos.
export function TeamBadge({ team, size = 40 }: { team: string; size?: number }) {
  const code = teamCode(team);
  const [showLogo, setShowLogo] = useState(TEAM_LOGOS_ENABLED);
  return (
    <span
      className={styles.badge}
      data-mode={showLogo ? "logo" : "crest"}
      style={{ "--team": teamColor(team), "--size": `${size}px` } as CSSProperties}
      title={team}
      aria-label={team}
    >
      {showLogo ? (
        <Image
          className={styles.logo}
          src={`/teams/${code}.png`}
          alt={team}
          width={size}
          height={size}
          unoptimized
          onError={() => setShowLogo(false)}
        />
      ) : (
        code
      )}
    </span>
  );
}
