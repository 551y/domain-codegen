import { TopNav } from "@/components/nav";

export default function GenerateLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <TopNav title="代码生成" />
      <main>{children}</main>
    </>
  );
}
