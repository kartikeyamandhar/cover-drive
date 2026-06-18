import styles from "./EventBurst.module.css";

// A broadcast-style flourish over the scoreboard on momentous deliveries. Rendered with a
// key on the ball id by the parent, so it remounts (and re-animates) once per delivery, then
// fades itself out via CSS. Returns nothing for routine balls.
function momentous(event: string): { label: string; kind: string } | null {
  if (event.startsWith("WICKET")) return { label: "WICKET", kind: "wicket" };
  if (event === "SIX off the bat") return { label: "SIX", kind: "six" };
  if (event === "FOUR off the bat") return { label: "FOUR", kind: "four" };
  return null;
}

export function EventBurst({ event }: { event: string }) {
  const m = momentous(event);
  if (!m) return null;
  return (
    <div className={styles.burst} data-kind={m.kind} aria-hidden="true">
      <span className={styles.label}>{m.label}</span>
    </div>
  );
}
