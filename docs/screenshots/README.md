# Dashboard screenshots

The three images referenced by the main [README](../../README.md) are not committed yet.
This folder is the drop-in location for them — capture once, and the image tags in the
README only need one line uncommented.

## Expected files

| File | What to capture | Suggested size |
| --- | --- | --- |
| `dashboard-dark.png` | Dark theme, page top: status panel, Current Flux / Predicted Flare Class / Confidence cards, forecast card, events table | 1440 × 900 or wider |
| `solar-flare-101.png` | The **Solar Flare 101** tab scrolled to the GOES scale bar and class table | 1440 × 900 |
| `mobile.png` | The same dashboard at phone width, everything stacked into one column | 390 × 844 |

## 60-second recipe

1. Start the app:
   ```bash
   streamlit run app.py
   ```
   then open <http://localhost:8501>.

2. Let the boot splash finish (about three seconds) and set **Theme → Dark** in the sidebar.
   A wide window matters: `1440 × 900` or larger shows the multi-column layout at its best.

3. Capture with **Win + Shift + S** (Windows) or **Cmd + Shift + 5** (macOS), or from
   Firefox/Chrome DevTools → *Capture screenshot* if you want the full page in one shot.

4. For `mobile.png`, open DevTools (`F12`) → toggle the device toolbar (`Ctrl + Shift + M`)
   → pick a phone preset such as *iPhone 12 Pro* (390 × 844) → capture, then reload the page
   so the layout re-flows at that width.

5. Save the files here with the exact names above, then uncomment the `<p align="center">`
   block inside `## Screenshots and artifacts` in the main README.

## Tips

- Hide the Streamlit toolbar for cleaner captures by adding a project config:
  ```toml
  # .streamlit/config.toml
  [client]
  toolbarMode = "minimal"
  ```
- Do not press `c` while the dashboard is focused — that is Streamlit's *clear cache*
  shortcut, not a way to hide the toolbar.
- Capture *after* the first prediction renders; on a cold start the cards briefly show
  the previous session's values while NOAA is being fetched.
