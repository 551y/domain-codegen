import { TopNav } from "@/components/nav";

export default function CompareLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <TopNav title="评测对比" />
      <main>{children}</main>
    </>
  );
}
