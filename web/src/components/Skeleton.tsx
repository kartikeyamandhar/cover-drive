import styles from "./Skeleton.module.css";

interface Props {
  width?: string;
  height?: string;
  radius?: string;
}

// A shimmer placeholder used instead of spinners while data loads.
export function Skeleton({ width = "100%", height = "1em", radius = "var(--r1)" }: Props) {
  return <span className={styles.skeleton} style={{ width, height, borderRadius: radius }} />;
}
