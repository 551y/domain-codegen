"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { navSections } from "@/config/site";
import { cn } from "@/lib/utils";

export default function Navigation() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-1 flex-col gap-y-5 overflow-y-auto px-3 py-4">
      {navSections.map((section) => (
        <div key={section.label}>
          <div className="px-2.5 pb-1.5 font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground/70">
            {section.label}
          </div>
          <div className="space-y-0.5">
            {section.items.map((navigation) => {
              const Icon = navigation.icon;
              const isActive =
                pathname === navigation.href ||
                (navigation.href !== "/" && pathname.startsWith(navigation.href));
              return (
                <Link
                  key={navigation.name}
                  href={navigation.href}
                  className={cn(
                    "group relative flex items-center rounded-lg px-2.5 py-2 text-[13.5px] transition-colors",
                    isActive
                      ? "bg-primary/10 font-medium text-primary dark:bg-primary/20"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <span
                    className={cn(
                      "absolute -left-3 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full bg-gradient-to-b from-primary to-accent transition-opacity",
                      isActive ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <Icon size={16} strokeWidth={1.8} className="mr-2.5 shrink-0" />
                  {navigation.name}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </nav>
  );
}
