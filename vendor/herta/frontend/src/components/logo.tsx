import { cn } from "@/lib/utils";

/**
 * Herta 品牌标识
 * 意象：倾斜轨道环 + 中心星体 + 轨道上运行的卫星 + 右上信号弧 —— 卫星通信协议。
 */
export default function Logo({ size = 36, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      className={cn("shrink-0", className)}
      role="img"
      aria-label="Herta"
    >
      <defs>
        <linearGradient id="herta-tile" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
          <stop stopColor="#2563eb" />
          <stop offset="1" stopColor="#0891b2" />
        </linearGradient>
      </defs>
      <rect x="1.5" y="1.5" width="45" height="45" rx="12" fill="url(#herta-tile)" />
      <rect x="1.5" y="1.5" width="45" height="45" rx="12" fill="white" opacity="0.06" />
      {/* 轨道环 */}
      <ellipse
        cx="23"
        cy="26"
        rx="15"
        ry="8.5"
        transform="rotate(-18 23 26)"
        stroke="#fff"
        strokeWidth="1.8"
        opacity="0.75"
      />
      {/* 中心星体 */}
      <circle cx="23" cy="26" r="4.6" fill="#fff" />
      {/* 轨道卫星 */}
      <circle cx="33.5" cy="19.4" r="2.5" fill="#fff" />
      {/* 信号弧 */}
      <path
        d="M35.6 15.2a5.8 5.8 0 0 1 3.1 3.1"
        stroke="#fff"
        strokeWidth="1.9"
        strokeLinecap="round"
        opacity="0.95"
      />
      <path
        d="M38.2 11.2a10.2 10.2 0 0 1 5.5 5.5"
        stroke="#fff"
        strokeWidth="1.9"
        strokeLinecap="round"
        opacity="0.5"
      />
    </svg>
  );
}
