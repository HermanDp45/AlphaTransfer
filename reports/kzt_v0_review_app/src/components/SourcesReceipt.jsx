import React, { useEffect, useId, useRef, useState } from "react";
import dataComposerIcon from "../../../../../assets/datascience-small.svg?url";

import { Icon } from "./Icon.jsx";
import { SourceInspector } from "./SourceInspector.jsx";

function useReceiptInputModality(rootRef) {
  useEffect(() => {
    const root = rootRef.current;
    const controller = new AbortController();
    const options = { capture: true, signal: controller.signal };
    // Listen before tab handlers and Escape focus restoration, including when
    // focus enters the Shadow DOM from elsewhere in the answer.
    const keyboardInput = (event) => {
      if (event.metaKey || event.altKey || event.ctrlKey || ["Shift", "Control", "Alt", "Meta"].includes(event.key)) return;
      root.dataset.inputModality = "keyboard";
    };
    root.ownerDocument.addEventListener("keydown", keyboardInput, options);
    // Tab can start in the host document and finish inside this iframe. Its
    // keyup reaches the newly focused receipt even when its keydown did not.
    root.ownerDocument.addEventListener("keyup", (event) => {
      if (event.key === "Tab") keyboardInput(event);
    }, options);
    root.ownerDocument.addEventListener("pointerdown", () => {
      root.dataset.inputModality = "pointer";
    }, options);
    return () => controller.abort();
  }, [rootRef]);
}

// The reference delays hover inspection, opens immediately from the keyboard,
// and keeps long identities readable without allowing a tooltip offscreen.
function useReceiptTooltips(rootRef, tooltipRef) {
  useEffect(() => {
    const root = rootRef.current;
    const tooltip = tooltipRef.current;
    const controller = new AbortController();
    const listen = (target, type, handler, options = {}) =>
      target.addEventListener(type, handler, { ...options, signal: controller.signal });
    const selector = "[data-source-title], [data-tooltip-text]";
    let active, hovered, focused, dismissed, timer;
    const hide = () => {
      clearTimeout(timer);
      active?.removeAttribute("aria-describedby");
      active = null;
      tooltip.hidden = true;
      delete tooltip.dataset.visible;
    };
    const needsTooltip = (chip) => chip?.getClientRects().length &&
      (!chip.matches(".receipt-filter") || [...chip.children].some((part) => part.scrollWidth > part.clientWidth));
    const show = (chip) => {
      if (!needsTooltip(chip) || dismissed === chip) return;
      hide();
      active = chip;
      if (chip.dataset.sourceTitle) {
        tooltip.replaceChildren();
        tooltip.dataset.kind = "source";
        const add = (className, text) => {
          const part = tooltip.ownerDocument.createElement("span");
          part.className = className;
          part.textContent = text;
          tooltip.append(part);
        };
        add("receipt-tooltip-title", chip.dataset.sourceTitle);
        if (chip.dataset.sourceReason) add("receipt-tooltip-reason", chip.dataset.sourceReason);
      } else {
        delete tooltip.dataset.kind;
        tooltip.textContent = chip.dataset.tooltipText;
      }
      tooltip.style.left = "8px";
      tooltip.style.top = "8px";
      tooltip.hidden = false;
      chip.setAttribute("aria-describedby", tooltip.id);
      const anchor = chip.getBoundingClientRect();
      const bounds = tooltip.getBoundingClientRect();
      const width = document.documentElement.clientWidth;
      const height = window.innerHeight;
      tooltip.style.left = `${Math.max(8, Math.min(anchor.left + (anchor.width - bounds.width) / 2, width - bounds.width - 8))}px`;
      const above = anchor.top - bounds.height - 8;
      tooltip.style.top = `${Math.max(8, above >= 8 ? above : Math.min(anchor.bottom + 8, height - bounds.height - 8))}px`;
      tooltip.dataset.visible = "true";
    };
    const schedule = (chip, delay = 700) => {
      clearTimeout(timer);
      if (chip === active || chip === dismissed) return;
      hide();
      if (needsTooltip(chip)) timer = setTimeout(() => show(chip), delay);
    };
    const dismissOnEscape = (event) => {
      if (event.key !== "Escape") return;
      const target = active || hovered || focused;
      if (target) { event.preventDefault(); event.stopPropagation(); dismissed = target; hide(); }
    };
    listen(root, "pointerover", (event) => {
      const chip = event.target.closest(selector);
      if (event.pointerType === "touch" || !chip || chip.contains(event.relatedTarget)) return;
      hovered = chip; dismissed = null; schedule(chip);
    });
    listen(root, "pointerout", (event) => {
      const chip = event.target.closest(selector);
      if (!chip || chip.contains(event.relatedTarget)) return;
      hovered = null;
      clearTimeout(timer);
      timer = setTimeout(() => { if (!tooltip.matches(":hover") && focused !== chip) hide(); }, 100);
    });
    listen(root, "focusin", (event) => {
      focused = event.target.closest(selector);
      if (focused) { dismissed = null; schedule(focused, 0); }
    });
    listen(root, "focusout", (event) => {
      const chip = event.target.closest(selector);
      focused = null;
      if (hovered !== chip) hide();
    });
    listen(root, "pointerdown", (event) => {
      const chip = event.target.closest(selector);
      if (event.pointerType === "touch" && needsTooltip(chip)) chip.dataset.touchInspect = active === chip ? "open" : "preview";
    });
    listen(root, "click", (event) => {
      const chip = event.target.closest(selector);
      if (!chip) { hide(); return; }
      if (chip.dataset.touchInspect === "preview") { event.preventDefault(); dismissed = null; show(chip); }
      else if (!chip.matches("a[href]")) { dismissed = null; show(chip); }
      delete chip.dataset.touchInspect;
    });
    listen(root, "keydown", (event) => {
      if (event.key === "Escape") {
        dismissOnEscape(event);
        return;
      }
      const chip = event.target.closest(selector);
      if (chip && ["Enter", " "].includes(event.key) && !chip.matches("a[href]")) {
        event.preventDefault(); dismissed = null; show(chip);
      }
    }, { capture: true });
    // Hover can start while keyboard focus is outside the Shadow-DOM receipt.
    listen(root.ownerDocument, "keydown", dismissOnEscape);
    listen(tooltip, "pointerenter", () => clearTimeout(timer));
    listen(tooltip, "pointerleave", () => { if (!hovered && !focused) hide(); });
    listen(root, "scroll", (event) => { if (event.target !== tooltip) hide(); }, { capture: true });
    listen(window, "scroll", (event) => { if (event.target !== tooltip) hide(); }, { capture: true });
    listen(window, "resize", hide);
    return () => { hide(); controller.abort(); };
  }, [rootRef, tooltipRef]);
}

function SourceCard({ item, collapsible }) {
  const [open, setOpen] = useState(!collapsible);
  const contentId = `receipt-card-${useId()}`;
  const toggle = useRef(null);
  function collapse() {
    setOpen(false);
    window.requestAnimationFrame(() => toggle.current?.focus());
  }
  return <article className="receipt-card" data-open={open} aria-label={item.title}
    onKeyDown={(event) => {
      if (collapsible && open && event.key === "Escape" && !event.defaultPrevented) {
        event.preventDefault(); event.stopPropagation(); collapse();
      }
    }}>
    {collapsible && <div className="receipt-card-label" hidden={open} aria-hidden="true">{item.title}</div>}
    {collapsible && <button ref={toggle} type="button" className="receipt-card-toggle"
      aria-label={open ? `Collapse ${item.title}` : item.title}
      aria-expanded={open} aria-controls={contentId} onClick={open ? collapse : () => setOpen(true)}>
      <Icon name="chevronDown" size={16} />
    </button>}
    <div id={contentId} hidden={!open} className="receipt-card-content">
      <SourceInspector component={{ id: item.id, title: item.title, description: item.description, kind: "narrative" }}
        receiptQueries={item.queries} filters={[]} allowCopy={false}
        onReceiptCollapse={collapsible ? collapse : undefined} receiptContentId={contentId}
        externalReceiptToggle={collapsible} />
    </div>
  </article>;
}

/** One answer-level disclosure; native answer prose is intentionally not part of this payload. */
export function SourcesReceipt({ items }) {
  const root = useRef(null);
  const tooltip = useRef(null);
  const tooltipId = `receipt-tooltip-${useId()}`;
  useReceiptInputModality(root);
  useReceiptTooltips(root, tooltip);
  const [open, setOpen] = useState(false);
  const contentId = `receipt-${useId()}`;
  const toggle = useRef(null);
  return <div ref={root} className="sources-receipt" data-input-modality="pointer" onKeyDown={(event) => {
    if (open && event.key === "Escape" && !event.defaultPrevented) {
      event.preventDefault(); event.stopPropagation(); setOpen(false); toggle.current?.focus();
    }
  }}>
    <div className="receipt-toolbar"><button ref={toggle} type="button" className="receipt-disclosure" aria-expanded={open}
      aria-controls={contentId} onClick={() => setOpen(!open)}>
      <img src={dataComposerIcon} width={16} height={16} alt="" aria-hidden="true" />
      <span>{items.length === 1 ? "Sources" : `Sources • ${items.length}`}</span>
      <Icon name="chevronDown" size={14} />
    </button></div>
    <div id={contentId} className="receipt-expander" data-open={open} inert={!open} aria-hidden={!open}>
      <div className="receipt-expander-inner"><div className="receipt-cards">
        {items.map((item) => <SourceCard key={item.id} item={item} collapsible={items.length > 1} />)}
      </div></div>
    </div>
    <div ref={tooltip} id={tooltipId} className="receipt-tooltip" role="tooltip" hidden />
  </div>;
}
