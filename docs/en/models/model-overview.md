---
title: Model Overview
hide:
    - toc
---

<!--
Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
-->

<div id="model-overview" class="md-typeset">
  <div class="mo-toolbar">
    <label class="mo-search">
      <input id="mo-search" type="search" placeholder="Search Models" aria-label="Search Models" />
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 5 1.5-1.5-5-5ZM9.5 14C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14Z"/></svg>
    </label>
    <div class="mo-filters" role="tablist" aria-label="Task">
      <span class="mo-filter-label">Task:</span>
      <button class="mo-chip is-active" data-task="all" role="tab">All</button>
      <button class="mo-chip" data-task="Detection" role="tab">Detection</button>
      <button class="mo-chip" data-task="Segmentation" role="tab">Segmentation</button>
      <button class="mo-chip" data-task="Classification" role="tab">Classification</button>
      <button class="mo-chip" data-task="Pose" role="tab">Pose</button>
      <button class="mo-chip" data-task="OBB" role="tab">OBB</button>
      <button class="mo-chip" data-task="Open Vocabulary" role="tab">Open Vocabulary</button>
    </div>
  </div>

  <div id="mo-content" class="mo-content" aria-live="polite"></div>
</div>

<link rel="stylesheet" href="../../stylesheets/model-overview.css">

<!-- Modal root -->
<div id="mo-modal-backdrop" class="mo-modal-backdrop" aria-hidden="true">
  <div class="mo-modal" role="dialog" aria-modal="true" aria-labelledby="mo-modal-title">
    <div class="mo-modal__header">
      <div class="mo-modal__title-row">
        <div class="mo-modal__title" id="mo-modal-title"></div>
        <div class="mo-modal__actions">
          <a id="mo-modal-details" class="mo-modal__details" href="#" target="_self">Details</a>
          <button class="mo-modal__close" type="button" aria-label="Close dialog">✕</button>
        </div>
      </div>
      <div class="mo-tablist" role="tablist" aria-label="Model details tabs">
        <button class="mo-tab" role="tab" aria-selected="true" data-tab="overview" id="mo-tab-overview">Overview</button>
        <button class="mo-tab" role="tab" aria-selected="false" data-tab="quick" id="mo-tab-quick">Quick Start</button>
        <button class="mo-tab" role="tab" aria-selected="false" data-tab="deploy" id="mo-tab-deploy">Deployment Options</button>
      </div>
    </div>
    <div class="mo-modal__description" id="mo-modal-description"></div>
    <div class="mo-modal__panel is-active" id="mo-panel-overview" role="tabpanel" aria-labelledby="mo-tab-overview"></div>
    <div class="mo-modal__panel" id="mo-panel-quick" role="tabpanel" aria-labelledby="mo-tab-quick"></div>
    <div class="mo-modal__panel" id="mo-panel-deploy" role="tabpanel" aria-labelledby="mo-tab-deploy"></div>
  </div>
</div>

<script src="../../javascript/model-overview.js"></script>
