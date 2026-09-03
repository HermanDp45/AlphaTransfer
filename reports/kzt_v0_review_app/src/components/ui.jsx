import * as DialogPrimitive from "@radix-ui/react-dialog";
import * as Dropdown from "@radix-ui/react-dropdown-menu";
import React, { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { Icon } from "./Icon.jsx";
import { dropdownModel } from "./dropdown-model.js";

export function Button({ variant = "secondary", className = "", ...props }) {
  return <button type="button" {...props} className={`button ${variant} ${className}`.trim()} />;
}

export function Tooltip({ children, className = "", portal = false, visible, ...props }) {
  const tooltip = (
    <span
      {...props}
      className={["dashboard-tooltip", className].filter(Boolean).join(" ")}
      data-visible={visible == null ? undefined : String(visible)}
      role="tooltip"
    >
      {children}
    </span>
  );
  return portal && typeof document !== "undefined" ? createPortal(tooltip, document.body) : tooltip;
}

/** Reveal clipped text without adding tooltips to labels that already fit. */
export function TruncatedText({ as: Tag = "span", children, ...props }) {
  const anchor = useRef(null), tip = useRef(null), timer = useRef(null);
  const [text, setText] = useState(null);
  const id = useId();
  const hide = () => { clearTimeout(timer.current); setText(null); };
  const show = () => {
    clearTimeout(timer.current);
    const element = anchor.current;
    if (!element || element.isContentEditable || element.scrollWidth <= element.clientWidth) return;
    timer.current = setTimeout(() => setText(element.textContent), 350);
  };
  useEffect(() => () => clearTimeout(timer.current), []);
  useLayoutEffect(() => {
    if (!text || !tip.current) return;
    const position = () => {
      const bounds = anchor.current.getBoundingClientRect(), tooltip = tip.current;
      const view = anchor.current.ownerDocument.defaultView;
      tooltip.style.maxWidth = `${Math.min(360, view.innerWidth - 24)}px`;
      const size = tooltip.getBoundingClientRect();
      tooltip.style.left = `${Math.max(12, Math.min(view.innerWidth - size.width - 12, bounds.left))}px`;
      tooltip.style.top = `${Math.max(12, bounds.bottom + size.height + 8 < view.innerHeight - 12
        ? bounds.bottom + 8 : bounds.top - size.height - 8)}px`;
    };
    position();
    const view = anchor.current.ownerDocument.defaultView;
    view.addEventListener("resize", hide);
    view.addEventListener("scroll", hide, true);
    return () => { view.removeEventListener("resize", hide); view.removeEventListener("scroll", hide, true); };
  }, [text]);
  return <><Tag {...props} ref={anchor} aria-describedby={text ? id : props["aria-describedby"]}
    onPointerEnter={event => { props.onPointerEnter?.(event); show(); }}
    onPointerLeave={event => { props.onPointerLeave?.(event); hide(); }}
    onPointerDown={event => { props.onPointerDown?.(event); hide(); }}
    onFocus={event => { props.onFocus?.(event); show(); }} onBlur={event => { props.onBlur?.(event); hide(); }}
    onKeyDown={event => { props.onKeyDown?.(event); if (event.key === "Escape") hide(); }}>{children}</Tag>
    {text && createPortal(<span ref={tip} id={id} role="tooltip" data-tooltip-portal="true" className="info-tooltip truncated-text-tooltip">{text}</span>,
      anchor.current.getRootNode().host ? anchor.current.getRootNode() : anchor.current.ownerDocument.body)}
  </>;
}

export function Menu({ label, trigger, children, align = "end", side = "bottom", contentClassName = "", open, onOpenChange, contentProps = {} }) {
  return (
    <Dropdown.Root modal={false} open={open} onOpenChange={onOpenChange}>
      <Dropdown.Trigger asChild>{trigger}</Dropdown.Trigger>
      <Dropdown.Portal>
        <Dropdown.Content
          {...contentProps}
          aria-label={label}
          align={align}
          side={side}
          className={["popover", contentClassName].filter(Boolean).join(" ")}
          collisionPadding={12}
          sideOffset={7}
          onCloseAutoFocus={(event) => {
            contentProps.onCloseAutoFocus?.(event);
            if (document.activeElement?.closest?.('[data-permalink-target="true"]')) event.preventDefault();
          }}
        >
          {children}
        </Dropdown.Content>
      </Dropdown.Portal>
    </Dropdown.Root>
  );
}

export function MenuItem({
  icon,
  leading,
  danger = false,
  children,
  subtext,
  onSelect,
  className = "",
  href,
  target,
  rel,
  ...props
}) {
  const itemProps = {
    ...props,
    className: `menu-item${subtext ? " has-subtext" : ""}${danger ? " danger" : ""}${className ? ` ${className}` : ""}`,
  };
  const content = (
    <>
      {leading ?? (icon && <Icon name={icon} />)}
      {subtext ? (
        <span className="menu-item-copy">
          <span className="menu-item-label">{children}</span>
          <span className="menu-item-subtext">{subtext}</span>
        </span>
      ) : (
        <span className="menu-item-label">{children}</span>
      )}
    </>
  );

  if (href)
    return (
      <Dropdown.Item
        asChild
        onSelect={(event) => {
          onSelect?.(event);
          event.preventDefault();
        }}
      >
        <a
          {...itemProps}
          href={href}
          target={target}
          rel={rel ?? "noopener noreferrer"}
          onKeyDownCapture={(event) => {
            if (event.key === "Enter") event.stopPropagation();
          }}
        >
          {content}
        </a>
      </Dropdown.Item>
    );

  return (
    <Dropdown.Item {...itemProps} onSelect={onSelect}>
      {content}
    </Dropdown.Item>
  );
}

export function MenuSub({ icon, leading, label, children, contentClassName = "", triggerClassName = "" }) {
  return (
    <Dropdown.Sub>
      <Dropdown.SubTrigger className={["menu-item", triggerClassName].filter(Boolean).join(" ")}>
        {leading ?? (icon && <Icon name={icon} />)}
        <span className="menu-item-label">{label}</span>
        <Icon name="chevronRight" className="menu-chevron" />
      </Dropdown.SubTrigger>
      <Dropdown.Portal>
        <Dropdown.SubContent
          aria-label={label}
          className={["popover", "menu-sub-content", contentClassName].filter(Boolean).join(" ")}
          collisionPadding={12}
          sideOffset={6}
        >
          {children}
        </Dropdown.SubContent>
      </Dropdown.Portal>
    </Dropdown.Sub>
  );
}

export function MenuSeparator() {
  return <Dropdown.Separator className="menu-separator" />;
}

export function MenuGroup({ label, children }) {
  return (
    <Dropdown.Group>
      <Dropdown.Label className="menu-group-label">{label}</Dropdown.Label>
      {children}
    </Dropdown.Group>
  );
}

export function InfoTooltip({ label = "More information", children }) {
  const id = useId();
  const triggerRef = useRef(null);
  const tooltipRef = useRef(null);
  const [visible, setVisible] = useState(false);

  useLayoutEffect(() => {
    if (!visible || typeof window === "undefined") return undefined;
    const trigger = triggerRef.current;
    const wrapper = trigger?.closest(".info-wrap");
    const tooltip = tooltipRef.current;
    if (!trigger || !wrapper || !tooltip) return undefined;

    function positionTooltip() {
      const viewport = { left: 12, right: window.innerWidth - 12 };
      const componentBounds = wrapper.closest("[data-component-id]")?.getBoundingClientRect();
      const componentRange = componentBounds && {
        left: Math.max(viewport.left, componentBounds.left + 12),
        right: Math.min(viewport.right, componentBounds.right - 12),
      };
      const range = componentRange && componentRange.right - componentRange.left >= 180 ? componentRange : viewport;
      tooltip.style.setProperty("--info-tooltip-max-width", `${Math.floor(range.right - range.left)}px`);
      const anchor = trigger.getBoundingClientRect();
      const bounds = tooltip.getBoundingClientRect();
      const above = anchor.top - bounds.height - 8 >= 12;
      const side = above ? "top" : "bottom";
      const center = anchor.left + anchor.width / 2;
      const left = Math.max(range.left, Math.min(range.right - bounds.width, center - bounds.width / 2));
      const preferredTop = above ? anchor.top - bounds.height - 8 : anchor.bottom + 8;
      const top = Math.max(12, Math.min(window.innerHeight - bounds.height - 12, preferredTop));
      wrapper.dataset.tooltipSide = side;
      tooltip.dataset.tooltipSide = side;
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
      tooltip.style.opacity = "1";
      tooltip.style.visibility = "visible";
    }

    positionTooltip();
    window.addEventListener("resize", positionTooltip);
    window.addEventListener("scroll", positionTooltip, true);
    return () => {
      window.removeEventListener("resize", positionTooltip);
      window.removeEventListener("scroll", positionTooltip, true);
    };
  }, [visible, children]);

  function hideTooltip() {
    if (triggerRef.current?.matches(":focus")) return;
    setVisible(false);
  }

  return (
    <span className="info-wrap" onPointerEnter={() => setVisible(true)} onPointerLeave={hideTooltip}>
      <button
        ref={triggerRef}
        type="button"
        className="info"
        aria-label={label}
        aria-describedby={visible ? id : undefined}
        onFocus={() => setVisible(true)}
        onBlur={() => setVisible(false)}
        onKeyDown={(event) => {
          if (event.key === "Escape") event.currentTarget.blur();
        }}
      >
        <Icon name="info" size={16} />
      </button>
      {visible && typeof document !== "undefined" && createPortal(
        <span ref={tooltipRef} id={id} className="info-tooltip" data-tooltip-portal="true" role="tooltip">
          {children}
        </span>,
        document.body,
      )}
    </span>
  );
}

export function Select({
  label,
  value,
  choices,
  onChange,
  showLabel = false,
  allLabel = "All",
  formatChoice,
  triggerClassName = "filter-trigger",
  contentClassName = "",
  align = "start",
  modal = true,
  scrollToSelected = false,
  groups,
  multiple = false,
  portalContainer,
}) {
  const { display, displayedValue, isSelected, select } = dropdownModel({ value, multiple, allLabel, formatChoice });
  const optionGroups = groups ?? [{ choices }];

  return (
    <Dropdown.Root modal={modal}>
      <Dropdown.Trigger className={`${triggerClassName} select-trigger`} aria-label={label}
        onPointerEnter={event => {
          const valueLabel = event.currentTarget.querySelector(".select-value");
          event.currentTarget.title = valueLabel?.scrollWidth > valueLabel?.clientWidth ? valueLabel.textContent : "";
        }}>
        {showLabel && <span className="filter-label">{label}</span>}
        <span className="select-value">{displayedValue}</span>
        <Icon name="chevronDown" className="chevron" />
      </Dropdown.Trigger>
      <Dropdown.Portal container={portalContainer}>
        <Dropdown.Content
          align={align}
          className={["popover", "select-content", contentClassName].filter(Boolean).join(" ")}
          collisionPadding={12}
          sideOffset={7}
          onOpenAutoFocus={(event) => {
            if (!scrollToSelected) return;
            const content = event.currentTarget;
            requestAnimationFrame(() =>
              content.querySelector('[data-state="checked"]')?.scrollIntoView({ block: "center" }),
            );
          }}
        >
          {multiple ? optionGroups.map((group, index) => (
            <React.Fragment key={group.label ?? index}>
              {group.label && (
                <Dropdown.Label className="menu-group-label select-group-label">{group.label}</Dropdown.Label>
              )}
              {group.choices.map((choice) => {
                const checked = isSelected(choice);
                return <Dropdown.CheckboxItem className="menu-item" key={choice || "none"}
                  checked={checked} onCheckedChange={() => onChange?.(select(choice))}
                  onSelect={(event) => event.preventDefault()}>
                  <span className="menu-item-label">{display(choice)}</span>
                  <Dropdown.ItemIndicator className="menu-check"><Icon name="check" /></Dropdown.ItemIndicator>
                </Dropdown.CheckboxItem>;
              })}
            </React.Fragment>
          )) : <Dropdown.RadioGroup value={value} onValueChange={onChange}>
            {optionGroups.map((group, index) => (
              <React.Fragment key={group.label ?? index}>
                {group.label && (
                  <Dropdown.Label className="menu-group-label select-group-label">{group.label}</Dropdown.Label>
                )}
                {group.choices.map((choice) => (
                  <Dropdown.RadioItem className="menu-item" key={choice || "none"} value={choice}>
                    <span className="menu-item-label">{display(choice)}</span>
                    <Dropdown.ItemIndicator className="menu-check">
                      <Icon name="check" />
                    </Dropdown.ItemIndicator>
                  </Dropdown.RadioItem>
                ))}
              </React.Fragment>
            ))}
          </Dropdown.RadioGroup>}
        </Dropdown.Content>
      </Dropdown.Portal>
    </Dropdown.Root>
  );
}

function tabElementId(tabsId, tabId, kind) {
  return `${tabsId}-${kind}-${tabId}`;
}

// One measured underline moves between tabs; layout changes never rely on guessed widths.
export function TabIndicator({ navRef, value, inset = 0, motion }) {
  const indicatorRef = useRef(null);
  const previous = useRef(null);
  useLayoutEffect(() => {
    const indicator = indicatorRef.current;
    const nav = navRef.current ?? indicator?.parentElement;
    if (!nav || !indicator) return;
    const update = () => {
      const selected = nav.querySelector('[role="tab"][aria-selected="true"]');
      if (!selected) {
        indicator.hidden = true;
        return;
      }
      indicator.hidden = false;
      const bounds = selected.getBoundingClientRect();
      const parent = nav.getBoundingClientRect();
      const leftInset = selected.closest(".dashboard-tab-item") === nav.firstElementChild ? 0 : inset;
      const next = {
        x: bounds.left - parent.left + nav.scrollLeft + leftInset,
        width: Math.max(0, bounds.width - leftInset - inset),
      };
      if (motion) indicator.style.transition = window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "none" : `transform ${motion.duration}ms ${motion.easing}, width ${motion.duration}ms ${motion.easing}`;
      indicator.style.width = `${next.width}px`;
      indicator.style.transform = `translateX(${next.x}px)`;
      const old = previous.current;
      if (
        !motion && old &&
        (old.x !== next.x || old.width !== next.width) &&
        !window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ) {
        indicator.getAnimations?.().forEach((animation) => animation.cancel());
        indicator.animate(
          [
            { transform: `translateX(${old.x}px)`, width: `${old.width}px` },
            { transform: `translateX(${next.x}px)`, width: `${next.width}px` },
          ],
          { duration: 180, easing: "cubic-bezier(.22,1,.36,1)" },
        );
      }
      previous.current = next;
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(nav);
    for (const tab of nav.querySelectorAll('[role="tab"]')) observer.observe(tab);
    return () => observer.disconnect();
  }, [navRef, value, inset, motion]);
  return <span ref={indicatorRef} className="tab-active-indicator" aria-hidden="true" />;
}

export function Tabs({ id, label, items, value, onChange, className = "", variant = "underline", indicatorMotion }) {
  const generatedId = useId();
  const navRef = useRef(null);
  const tabsId = id ?? `tabs-${generatedId}`;
  return (
    <nav
      ref={navRef}
      className={`tabs ${variant === "underline" ? "source-tabs" : ""} ${className}`.trim()}
      role="tablist"
      aria-label={label}
      aria-orientation="horizontal"
      onKeyDown={(event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key) || !items.length) return;
        event.preventDefault();
        const current = Math.max(
          0,
          items.findIndex((item) => item.id === value),
        );
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? items.length - 1
              : (current + (event.key === "ArrowRight" ? 1 : -1) + items.length) % items.length;
        onChange(items[next].id);
        event.currentTarget.querySelectorAll("[role='tab']")[next]?.focus();
      }}
    >
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={value === item.id}
          id={tabElementId(tabsId, item.id, "tab")}
          aria-controls={tabElementId(tabsId, item.id, "panel")}
          tabIndex={value === item.id ? 0 : -1}
          onClick={() => onChange(item.id)}
        >
          {item.label}
        </button>
      ))}
      {(variant === "underline" || className.split(" ").includes("source-tabs")) && <TabIndicator navRef={navRef} value={value} motion={indicatorMotion} />}
    </nav>
  );
}

export function TabPanel({ tabsId, tabId, active, className = "", children, keepMounted = false }) {
  if (!active && !keepMounted) return null;
  return (
    <section
      className={`tab-panel ${className}`.trim()}
      role="tabpanel"
      id={tabElementId(tabsId, tabId, "panel")}
      aria-labelledby={tabElementId(tabsId, tabId, "tab")}
      hidden={!active || undefined}
      tabIndex={0}
    >
      {children}
    </section>
  );
}

export function Dialog({
  open = true,
  title,
  titleInfo,
  initialFocusSelector,
  expanded = false,
  className = "",
  style,
  headerActions,
  onClose,
  children,
  showClose = true,
  backdropClassName = "",
}) {
  const returnFocus = useRef(null);
  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className={["dialog-backdrop", backdropClassName].filter(Boolean).join(" ")} />
        <DialogPrimitive.Content
          style={style}
          data-shared-dialog
          aria-describedby={undefined}
          onOpenAutoFocus={(event) => {
            returnFocus.current = event.currentTarget.ownerDocument.activeElement;
            if (!initialFocusSelector) return;
            const initial = event.currentTarget.querySelector(initialFocusSelector);
            if (!initial) return;
            event.preventDefault();
            initial.focus({ preventScroll: true });
          }}
          onCloseAutoFocus={(event) => {
            const routeTarget = event.currentTarget.ownerDocument.querySelector('[data-permalink-target="true"]');
            // Navigation owns focus when it moves away from this dialog's opener.
            if (routeTarget && !routeTarget.contains(returnFocus.current)) {
              event.preventDefault();
              routeTarget.focus({ preventScroll: true });
              return;
            }
            if (!returnFocus.current?.isConnected) return;
            event.preventDefault();
            returnFocus.current.focus({ preventScroll: true });
          }}
          className={["dialog", expanded ? "expanded" : "source-dialog", className].filter(Boolean).join(" ")}
        >
          <header className="dialog-header" tabIndex={initialFocusSelector === ".dialog-header" ? -1 : undefined}>
            <div className="dialog-title-group">
              <DialogPrimitive.Title>{title}</DialogPrimitive.Title>
              {titleInfo && <InfoTooltip label="How scheduling works">{titleInfo}</InfoTooltip>}
            </div>
            <div className="dialog-header-actions">
              {headerActions}
              {showClose && (
                <DialogPrimitive.Close className="icon-button" aria-label="Close">
                  <Icon name="cross" size={20} />
                </DialogPrimitive.Close>
              )}
            </div>
          </header>
          <div className="dialog-content">{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
