import React, { useId } from "react";

import { publishReviewDestination } from "../publish-review.js";
import { dataAppPromptTarget } from "../runtime-environment.js";
import { Icon } from "./Icon.jsx";
import { Dialog } from "./ui.jsx";

export function PublishReviewContent({ published, surface, accessMode, onAccessChange, href, onClose }) {
  const id = useId();
  const report = surface === "report";
  const destination = publishReviewDestination(href);
  return <>
    <div className="publish-review-body">
      <section className="publish-review-section" aria-labelledby={`${id}-audience`}>
        <h3 id={`${id}-audience`}>Who can access</h3>
        {published ? <div className="publish-review-existing">
          <Icon name="lock" size={18} />
          <div><strong>Keep current access</strong><p>Publishing changes won’t change who can view.</p></div>
        </div> : <fieldset className="publish-review-audience" aria-labelledby={`${id}-audience`}>
          {[
            { value: "custom", icon: "lock", title: "Invited people", detail: "Only you until you invite others." },
            { value: "workspace_all", icon: "building", title: "Workspace members", detail: "Anyone in your workspace with the link." },
          ].map(option => <label key={option.value} className="publish-review-option">
            <Icon name={option.icon} size={18} />
            <span><strong>{option.title}</strong><span className="publish-review-option-detail">{option.detail}</span></span>
            <input type="radio" name={`${id}-access`} value={option.value} checked={accessMode === option.value}
              onChange={() => onAccessChange(option.value)} aria-label={option.title} />
          </label>)}
        </fieldset>}
      </section>

      <div className="publish-review-permissions">
        <ul className="publish-review-facts">
          <li>{report ? "Report" : "Dashboard"} data is stored in the Site.</li>
          <li>Viewers can view and download that data.</li>
          <li>Editors can change the Site.{!report && " Refreshing data requires their own source permissions."}</li>
        </ul>
        <p>Includes all saved rows, even those hidden by filters. Editing doesn’t grant source access.</p>
      </div>
    </div>
    <footer className="publish-review-footer">
      <button type="button" className="button" onClick={onClose}>Cancel</button>
      {href && destination ? <a className="button primary dashboard-publish-confirm" href={href} target={dataAppPromptTarget(href)} rel="noopener noreferrer">
        Publish in ChatGPT<Icon name="arrowUpRight" size={16} />
      </a> : <button type="button" className="button primary dashboard-publish-confirm" disabled>Publishing unavailable</button>}
    </footer>
  </>;
}

export function PublishReviewDialog({ open, onClose, ...props }) {
  const noun = props.surface === "report" ? "report" : "dashboard";
  return <Dialog open={open} onClose={onClose} title={props.published ? "Publish changes" : `Publish ${noun}`}
    className="publish-review-dialog" initialFocusSelector=".dialog-header">
    {open && <PublishReviewContent {...props} onClose={onClose} />}
  </Dialog>;
}
