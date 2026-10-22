import Logo from "@/components/logo";

export default function Brand() {
  return (
    <div className="flex items-center border-b border-border px-5 py-4">
      <div className="flex items-center gap-3">
        <Logo size={40} className="shadow-soft-md rounded-[10px]" />
        <div className="min-w-0">
          <div className="font-sans text-[15px] font-bold leading-tight tracking-tight">
            HERTA<span className="text-accent">.</span>
          </div>
          <div className="mt-0.5 truncate font-mono text-[10px] tracking-wide text-muted-foreground">
            PROTOCOL GENERATOR
          </div>
        </div>
      </div>
    </div>
  );
}
