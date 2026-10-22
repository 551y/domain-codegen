import Link from "next/link";
import Logo from "@/components/logo";

export default function NotFound() {
  return (
    <div className="flex min-h-[80vh] flex-col items-center justify-center px-8">
      <Logo size={56} className="rounded-2xl shadow-soft-lg" />
      <div className="mt-8 font-mono text-6xl font-bold tracking-tight text-gradient">404</div>
      <p className="mt-3 text-sm text-muted-foreground">页面不存在或已被移动</p>
      <Link
        href="/"
        className="mt-6 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground shadow-soft-md transition-shadow hover:shadow-soft-lg"
      >
        返回仪表盘
      </Link>
    </div>
  );
}
