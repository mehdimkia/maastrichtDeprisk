/* components/ui/tooltip.tsx — minimal Radix tooltip wrapper (no shadcn utils) */
"use client";

import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

function cn(...parts: Array<string | undefined>) {
  return parts.filter(Boolean).join(" ");
}

const TooltipProvider = TooltipPrimitive.Provider;
const Tooltip = TooltipPrimitive.Root;
const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 6, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-700 shadow-md",
      "dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200",
      className
    )}
    {...props}
  />
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
