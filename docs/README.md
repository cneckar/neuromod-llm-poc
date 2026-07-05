# PiHK.AI Website

This is the static website for PiHK.AI (Perturbagens I Have Known and Inferred), hosted on GitHub Pages.

## Files

- `index.html` - Main homepage with project overview and features
- `effects.html` - Erowid-style **Effect Index**: every low-level neuromodulation effect and exactly what it does inside the model runtime (forward hooks, logit processors, KV-cache surgery, MoE routing). Master/detail; data from `reference-data.js`.
- `packs.html` - Erowid-style **Pack Index**: every predefined "drug" recipe with strengths/directions, expected behavioral artifacts, dose-response findings, and probes to try. Master/detail; effects cross-link to `effects.html`.
- `reference-data.js` - `window.EFFECTS` / `window.PACKS` data backing the two index pages (grounded in `neuromod/effects.py` and `packs/config.json`).
- `chat.html` - Interactive chat embedding the live neuromodulation Worker (chat.pihk.ai)
- `README.md` - This file

## Features

### Homepage (`index.html`)
- Project overview and mission
- Feature highlights
- Quick start guide
- Research applications
- Statistics and metrics
- Community links

### Interactive Chat (`chat.html`)
- Embeds the live Cloudflare Worker (https://chat.pihk.ai) in an iframe
- Drag-and-drop pillbox + real steered generation on a live model
- Dose (intensity) control and per-tier model badge

## Deployment

The website is automatically deployed to GitHub Pages using the `.github/workflows/pages.yml` workflow.

## Local Development

To run the website locally:

```bash
# Simple HTTP server
python -m http.server 8000

# Or with Node.js
npx serve docs/website

# Or with any static file server
cd docs/website
# Open index.html in browser
```

## Customization

The website uses:
- **CSS Grid** for responsive layouts
- **CSS Animations** for smooth interactions
- **Vanilla JavaScript** for interactivity
- **Google Fonts** (Inter) for typography
- **Gradient backgrounds** for visual appeal

## Design Principles

1. **Scientific Rigor** - Professional appearance suitable for academic use
2. **Playful Elements** - Fun, engaging design that reflects the experimental nature
3. **Accessibility** - Clear typography, good contrast, responsive design
4. **Performance** - Lightweight, fast-loading static site
5. **Mobile-First** - Responsive design that works on all devices

## Color Scheme

- **Primary**: `#667eea` (Blue-purple)
- **Secondary**: `#764ba2` (Purple)
- **Accent**: `#a8edea` (Light blue-green)
- **Text**: `#333` (Dark gray)
- **Background**: `#f8f9fa` (Light gray)

## Typography

- **Font**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700
- **Sizes**: Responsive scaling from 0.9rem to 3.5rem

## Future Enhancements

- [ ] Add more interactive demos
- [ ] Include real API integration
- [ ] Add blog section for research updates
- [ ] Implement dark mode
- [ ] Add more animations and micro-interactions
- [ ] Include video demonstrations
- [ ] Add accessibility improvements
