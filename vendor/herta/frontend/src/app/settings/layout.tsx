import { TopNav } from "@/components/nav";

export default function SettingsLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <TopNav title="设置" />
      <main>{children}</main>
    </>
  );
}
