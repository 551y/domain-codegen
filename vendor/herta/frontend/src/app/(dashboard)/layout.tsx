import { TopNav } from "@/components/nav";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <TopNav title="仪表盘" />
      <main>{children}</main>
    </>
  );
}
