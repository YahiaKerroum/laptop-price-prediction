"use client";

import { animate, motion, useInView, useReducedMotion } from "motion/react";
import { useEffect, useId, useMemo, useRef, useState } from "react";

export const spring = { type: "spring", stiffness: 260, damping: 30, mass: 0.9 } as const;

/** A number that springs from its previous value to the new one. */
export function Roll({ value, format }: { value: number; format: (v: number) => string }) {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  const reduce = useReducedMotion();
  useEffect(() => {
    if (reduce) {
      from.current = value;
      setShown(value);
      return;
    }
    const controls = animate(from.current, value, {
      duration: 0.9,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => {
        from.current = v;
        setShown(v);
      },
    });
    return () => controls.stop();
  }, [value, reduce]);
  return <span className="num">{format(shown)}</span>;
}

/** Counts up from zero the first time it scrolls into view. */
export function CountUp({ value, format }: { value: number; format: (v: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useInView(ref, { once: true, margin: "-60px" });
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!seen) return;
    if (reduce) return setShown(value);
    const c = animate(0, value, { duration: 1.6, ease: [0.16, 1, 0.3, 1], onUpdate: setShown });
    return () => c.stop();
  }, [seen, value, reduce]);
  return (
    <span ref={ref} className="num">
      {format(shown)}
    </span>
  );
}

export function Segmented<T extends string | number | null>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  const id = useId();
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button key={String(o.value)} type="button" aria-pressed={on} onClick={() => onChange(o.value)}>
            {on && <motion.span layoutId={`thumb-${id}`} className="thumb" transition={spring} />}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

type Item = { value: string; label: string; meta?: string };

/** Type-to-search picker over a few hundred options; keyboard friendly. */
export function Combo({
  items,
  value,
  onChange,
  placeholder,
  id,
}: {
  items: Item[];
  value: string | null;
  onChange: (v: string | null) => void;
  placeholder: string;
  id: string;
}) {
  const selected = items.find((i) => i.value === value) ?? null;
  const [query, setQuery] = useState(selected?.label ?? "");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => setQuery(selected?.label ?? ""), [selected?.label]);

  const matches = useMemo(() => {
    const terms = query.toLowerCase().split(/[\s-]+/).filter(Boolean);
    if (!open || (selected && query === selected.label)) return items.slice(0, 60);
    return items.filter((i) => terms.every((t) => i.label.toLowerCase().replace(/-/g, " ").includes(t) || i.label.toLowerCase().includes(t))).slice(0, 60);
  }, [items, query, open, selected]);

  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const pick = (item: Item) => {
    onChange(item.value);
    setQuery(item.label);
    setOpen(false);
  };

  return (
    <div className="combo">
      <input
        id={id}
        className="input"
        role="combobox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-autocomplete="list"
        autoComplete="off"
        placeholder={placeholder}
        value={query}
        onFocus={(e) => {
          setOpen(true);
          e.currentTarget.select();
        }}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          setActive(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setOpen(true);
            setActive((a) => Math.min(a + 1, matches.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((a) => Math.max(a - 1, 0));
          } else if (e.key === "Enter" && open && matches[active]) {
            e.preventDefault();
            pick(matches[active]);
          } else if (e.key === "Escape") {
            setOpen(false);
          }
        }}
        style={{ paddingRight: value ? "2.6rem" : undefined }}
      />
      {value && (
        <button
          type="button"
          className="combo-clear"
          aria-label="Clear"
          onClick={() => {
            onChange(null);
            setQuery("");
          }}
        >
          ×
        </button>
      )}
      {open && matches.length > 0 && (
        <ul className="combo-list" id={`${id}-list`} role="listbox" ref={listRef}>
          {matches.map((m, i) => (
            <li
              key={m.value}
              data-i={i}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault();
                pick(m);
              }}
            >
              <span>{m.label}</span>
              {m.meta && <span className="small">{m.meta}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
