import { TopNav } from "@/components/nav";

export default function KnowledgeLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <TopNav title="知识库" />
      <main>{children}</main>
    </>
  );
}
