import { cn } from "@/lib/utils";

/**
 * OpAlign 品牌标识
 * 意象：左侧两个长度不一的算子块（语义参差）经箭头转换后右侧对齐等长，
 * 底部虚线为对齐基准线 —— 对应“算子语义对齐”。
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
      aria-label="OpAlign"
    >
      <defs>
        <linearGradient id="oa-tile" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
          <stop stopColor="#2563eb" />
          <stop offset="1" stopColor="#0891b2" />
        </linearGradient>
      </defs>
      <rect x="1.5" y="1.5" width="45" height="45" rx="12" fill="url(#oa-tile)" />
      <rect x="1.5" y="1.5" width="45" height="45" rx="12" fill="white" opacity="0.06" />
      {/* 左：参差算子块 */}
      <rect x="9" y="13" width="11" height="5" rx="2.5" fill="#fff" />
      <rect x="9" y="21.5" width="6.5" height="5" rx="2.5" fill="#fff" opacity="0.55" />
      {/* 中：转换箭头 */}
      <path
        d="M21.5 19.75h6.2m0 0-2.4-2.4m2.4 2.4-2.4 2.4"
        stroke="#fff"
        strokeWidth="2.1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* 右：对齐算子块 */}
      <rect x="29.5" y="13" width="9.5" height="5" rx="2.5" fill="#fff" />
      <rect x="29.5" y="21.5" width="9.5" height="5" rx="2.5" fill="#fff" />
      {/* 底部对齐基准线 */}
      <rect x="9" y="31.5" width="30" height="2.2" rx="1.1" fill="#fff" opacity="0.45" />
      <rect x="9" y="31.5" width="13" height="2.2" rx="1.1" fill="#fff" opacity="0.9" />
    </svg>
  );
}
