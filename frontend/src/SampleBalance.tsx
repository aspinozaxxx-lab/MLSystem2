import { type CSSProperties, useEffect, useRef, useState } from "react";
import { editSamplePercentage, moveSampleBoundary, type SamplePercentages } from "./utils/trainingLaunch";

const ZONE_LABELS = ["С объектами", "Hard negative", "Фон"];

export function SampleBalance({ value, onChange }: { value: SamplePercentages; onChange: (value: SamplePercentages) => void }) {
  const track = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [draft, setDraft] = useState<{ index: number; text: string } | null>(null);
  useEffect(() => {
    const element = track.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const edges: [number, number] = [value[0], value[0] + value[1]];
  const centers = [value[0] / 2, value[0] + value[1] / 2, 100 - value[2] / 2].map((percent) => width * percent / 100);
  const lifted = value.map((percent) => width * percent / 100 < 76);
  const labelPositions = [...centers];
  const liftedIndices = lifted.flatMap((lift, index) => lift ? [index] : []);
  liftedIndices.forEach((index, i) => {
    labelPositions[index] = Math.max(38, centers[index], i ? labelPositions[liftedIndices[i - 1]] + 80 : 0);
  });
  [...liftedIndices].reverse().forEach((index, i, indices) => {
    labelPositions[index] = Math.min(width - 38, labelPositions[index], i ? labelPositions[indices[i - 1]] - 80 : width);
  });
  const closeHandles = (edges[1] - edges[0]) * width / 100 < 24;

  const drag = (boundary: 0 | 1, clientX: number) => {
    const rect = track.current?.getBoundingClientRect();
    if (rect?.width) onChange(moveSampleBoundary(value, boundary, Math.round((clientX - rect.left) / rect.width * 100)));
  };

  return (
    <section className="sample-balance" aria-label="Баланс выборки">
      <div className="sample-balance-heading"><h3>Баланс выборки</h3><span>Сумма — 100%</span></div>
      <div className={`sample-balance-shell${lifted.some(Boolean) ? " has-lifted-values" : ""}`}>
        <div className="sample-balance-track" ref={track}>
          <div className="sample-balance-fill" aria-hidden="true">
            {value.map((percent, index) => <div key={index} className={`sample-zone sample-zone-${index}`} style={{ width: `${percent}%` }} />)}
          </div>
          {value.map((percent, index) => (
            <label key={index} className={`sample-value sample-zone-${index}${lifted[index] ? " is-lifted" : ""}`}
              style={{ left: `${labelPositions[index] / width * 100}%`, "--sample-anchor": `${centers[index] - labelPositions[index]}px` } as CSSProperties}>
              <input type="number" min="0" max="100" step="any" required aria-label={`${ZONE_LABELS[index]}, процент`}
                value={draft?.index === index ? draft.text : percent}
                onFocus={() => setDraft({ index, text: String(percent) })}
                onChange={(event) => {
                  const raw = event.target.value;
                  setDraft({ index, text: raw });
                  const next = Number(raw);
                  if (raw !== "" && Number.isFinite(next) && next >= 0 && next <= 100) onChange(editSamplePercentage(value, index, next));
                }}
                onBlur={() => setDraft(null)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
                }}
              />
              <span aria-hidden="true">%</span>
            </label>
          ))}
          {edges.map((edge, index) => {
            const boundary = index as 0 | 1;
            const min = boundary === 0 ? 0 : edges[0];
            const max = boundary === 0 ? edges[1] : 100;
            return (
              <div key={boundary} className="sample-divider" style={{ left: `${edge}%` }}>
                <button type="button" role="slider" className={`sample-handle${closeHandles ? ` is-stacked-${boundary}` : ""}`}
                  aria-label={boundary === 0 ? "Граница объектов и hard negative" : "Граница hard negative и фона"}
                  aria-orientation="horizontal" aria-valuemin={min} aria-valuemax={max} aria-valuenow={edge}
                  aria-valuetext={`${edge}%`}
                  onPointerDown={(event) => {
                    if (event.button !== 0) return;
                    event.preventDefault();
                    event.currentTarget.focus();
                    event.currentTarget.setPointerCapture(event.pointerId);
                  }}
                  onPointerMove={(event) => {
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) drag(boundary, event.clientX);
                  }}
                  onPointerUp={(event) => {
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                      drag(boundary, event.clientX);
                      event.currentTarget.releasePointerCapture(event.pointerId);
                    }
                  }}
                  onKeyDown={(event) => {
                    const step = event.shiftKey ? 10 : 1;
                    const next = { ArrowLeft: edge - step, ArrowDown: edge - step, ArrowRight: edge + step, ArrowUp: edge + step, Home: min, End: max }[event.key];
                    if (next != null) { event.preventDefault(); onChange(moveSampleBoundary(value, boundary, next)); }
                  }}
                >
                  <span aria-hidden="true">Ⅱ</span>
                </button>
              </div>
            );
          })}
        </div>
      </div>
      <p className="sample-balance-legend">с объектами — hard negative — фон</p>
    </section>
  );
}
