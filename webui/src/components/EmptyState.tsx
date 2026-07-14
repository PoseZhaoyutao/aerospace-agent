import { Sparkles } from "lucide-react";
import { Composer } from "./Composer";

type Props = { disabled?: boolean; onSubmit: (message: string) => void };

export function EmptyState({ disabled, onSubmit }: Props) {
  return (
    <section className="empty-state" data-layout="empty-state">
      <div className="eyebrow"><Sparkles size={15} /> FLIGHT DESK / READY</div>
      <h1>What should we<br /><em>tackle together?</em></h1>
      <p className="empty-copy">A quiet workspace for complex aerospace reasoning, one verified thread at a time.</p>
      <Composer disabled={disabled} onSubmit={onSubmit} />
    </section>
  );
}
