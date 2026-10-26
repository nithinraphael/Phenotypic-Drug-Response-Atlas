---
name: Phenotypic Drug-Response Atlas
description: Scientific figure plates for measured responses, morphology and conditioned predictions.
colors:
  ink: "#163c3b"
  muted: "#526764"
  measured: "#176b62"
  prediction: "#315bbb"
  fitted-span: "#b66b13"
  canvas: "#f6f8f7"
  surface: "#ffffff"
  border: "#dce5e1"
  unavailable: "#eef3f1"
  plot-grid: "#e8eeeb"
  baseline: "#738984"
  fit-quality-light: "#b6d7c9"
  selection: "#d4e8e1"
typography:
  display:
    fontFamily: '"PT Sans", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "2rem"
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: "-0.025em"
  headline:
    fontFamily: '"PT Sans", system-ui, sans-serif'
    fontSize: "1.5rem"
    fontWeight: 600
    letterSpacing: "-0.015em"
  title:
    fontFamily: '"PT Sans", system-ui, sans-serif'
    fontSize: "1.125rem"
    fontWeight: 600
  body:
    fontFamily: '"PT Sans", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "16px"
    lineHeight: 1.55
  caption:
    fontSize: "0.875rem"
    lineHeight: 1.6
  metric:
    fontSize: "1.75rem"
    fontWeight: 600
    lineHeight: 1.3
  plot:
    fontFamily: '"PT Sans", system-ui, sans-serif'
    fontSize: "14px"
  tab:
    fontSize: "15px"
rounded:
  field: "8px"
  panel: "12px"
spacing:
  small: "8px"
  control: "16px"
  group: "24px"
components:
  metric-card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.panel}"
    padding: "18px 20px"
  unavailable-panel:
    backgroundColor: "{colors.unavailable}"
    rounded: "{rounded.panel}"
    padding: "24px"
  native-select:
    rounded: "{rounded.field}"
  tab-selected:
    textColor: "{colors.measured}"
    height: "44px"
---

# Design System: Phenotypic Drug-Response Atlas

## Overview

**Creative North Star: "Journal Figure Plates"**

Journal figure plates guide the atlas: pale canvas, dark teal text and white plotting surfaces. PT Sans carries headings, controls and scientific labels. Figures have room for axes, units and legends; compact native controls keep the working conditions visible.

The interface uses marimo tabs, selectors, tables and disclosures with a small CSS layer. Existing microscopy supplies the imagery. Color, symbols, line styles and captions distinguish observations, fitted responses and conditioned predictions.

**Key Characteristics:**

- White figure plates on a pale scientific canvas.
- Native controls and responsive comparison layouts.
- Explicit scientific labels and restrained evidence colors.

## Colors

A restrained teal palette supports the working interface; evidence colors retain specific scientific roles.

### Primary

- **Measured teal:** observations, Hill fits and selected tab text.
- **Prediction blue:** conditioned model curves and keyboard focus.
- **Fitted-span amber:** the labeled fitted effect span.

### Neutral

- **Deep teal ink / muted teal:** primary text and secondary descriptions.
- **Scientific canvas / figure white:** page ground and plotting surfaces.
- **Quiet border / plot grid:** panel boundaries and quantitative guides.
- **Unavailable wash:** explanatory panels for unsupported evidence.
- **Baseline gray-green:** comparison models other than Hill and TabPFN.
- **Fit-quality light teal:** the low end of the observed R² color scale.
- **Selection wash:** selected text.

**The Evidence Color Rule.** Pair evidence color with a legend label and, for response curves, marker or line style. Amber denotes a fitted effect span, never uncertainty.

## Typography

**Display Font:** PT Sans with system sans-serif fallbacks.  
**Body Font:** PT Sans with the same fallbacks.

One family keeps scientific identifiers, prose and chart labels visually related. Heading weight and size establish hierarchy without decorative typography.

### Hierarchy

- **Display:** page title; mobile reduces to (1.65rem).
- **Headline:** section headings.
- **Title:** subsection headings and unavailable-state titles.
- **Body:** descriptions and controls; header prose uses (1.65) line height and a (72ch) limit.
- **Caption:** secondary descriptions, up to (75ch).
- **Metric:** tabular numeric figures; mobile reduces to (1.4rem).
- **Plot:** chart text, with (20px) general titles, (18px) response titles and (12px) legends.
- **Tab:** navigation, reduced to (14px) on mobile; selected tabs use weight (600).

## Layout

The centered notebook shell has a maximum width of (1248px); marimo supplies its content gutters. Figures use the available content width and a standard height of (510px). Native vertical stacks carry section spacing; there is no universal 40px section token in the implementation.

Control groups use an auto-fitting grid with a (240px) minimum column, bounded by available width, and the control spacing token. Metrics have four columns. Microscopy comparison has two equal columns with the group gap.

At (700px) and below, navigation becomes a two-column grid, metrics become two columns with (10px) gaps and (14px) internal padding, and microscopy becomes one column with (20px) gaps. Controls stack as their available width requires. Native tables retain pagination and horizontal access to wide data.

## Elevation & Depth

White figure surfaces, quiet borders and pale unavailable panels establish depth. Custom atlas CSS adds no box shadows or motion. Native marimo widgets retain their own state styling, including the selected-tab surface visible in the screenshots; this is runtime styling, not a separate atlas shadow token.

## Shapes

Panels and microscopy images share the panel radius. Selects use the smaller field radius. Scientific plots remain rectangular white plates. Metric and unavailable panels use a (1px) quiet border; there are no decorative frames.

## Components

### Buttons

Native marimo run and download actions retain their supplied appearance and behavior. The CSS sets a minimum height of (42px) and text size of (16px) for button elements outside tabs. Download widgets may render as links and retain their native sizing. Keyboard focus uses a blue (2px) outline with (3px) offset.

### Cards / Containers

Metric cards use white surfaces, a quiet border, panel radius and tabular values. Unavailable panels use the pale wash and a brief heading plus explanation. These panels describe evidence limits rather than filling gaps with fabricated visuals.

### Inputs / Fields

Native selectors and searchable dropdowns keep their built-in interaction and labels. Select elements receive the field radius and horizontal padding of (10px). Inputs and selects have a minimum height of (42px) and text size of (16px). The same visible focus outline applies; no custom disabled or error palette is defined.

**The Native Control Rule.** Keep marimo interaction behavior; apply the atlas type, sizing and focus treatment without rebuilding the widget.

### Navigation

Four native tabs expose Discoveries, Drug Explorer, Cell Explorer and Validation. Tabs have a minimum height of (44px); the active label uses measured teal and weight (600). Mobile places the tabs in two rows with (10px) padding. Native hover and selected surfaces remain marimo-owned.

### Figure Plates

Plotly white surfaces carry explicit axes and units, light grids and horizontal legends below the plot. Measured viability uses teal circles; the Hill fit uses a solid teal line; the conditioned TabPFN prediction uses dashed blue. The fitted effect span is labeled separately. Response concentration remains logarithmic. Charts support native interaction and SVG export.

### Scientific Images and Tables

Existing microscopy images fill their columns without changing aspect ratio. Captions retain condition and field identity, control matching limits and independent display contrast. Native tables provide row selection, search, export and pagination. Methods and interpretation use native accordions.

## Do's and Don'ts

### Do:

- **Do** retain native marimo controls and visible keyboard focus.
- **Do** label axes, units, conditions and evidence types in figures and captions.
- **Do** keep microscopy provenance and display adjustments visible.
- **Do** distinguish measurements, fits, conditioned predictions and fitted effect spans.

### Don't:

- **Don't** present fitted effect spans as confidence intervals or biological tipping points.
- **Don't** imply PRISM viability and BBBC021 microscopy are paired assays.
- **Don't** add decorative imagery or custom JavaScript to this interface.
- **Don't** manufacture unavailable uncertainty, embeddings or external comparisons.

## Step 18.1 restored states

The existing design remains unchanged. The image-field map uses a two-line model/condition title, native Plotly color legends and a 600px grid wrapper to preserve interactive bounds across desktop/mobile resize. Points link to the exact microscopy field; captions identify the ImageNet baseline and unavailable Cell-DINO evidence. Prediction exports sit beside measured-observation downloads with Prior Labs attribution. The external source inventory remains inside a disclosure beside explicit comparison blockers.
