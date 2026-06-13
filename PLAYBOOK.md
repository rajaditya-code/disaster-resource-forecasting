# Resource Allocation Playbook
## How an NGO Can Operationalise These Forecasts

---

### Who Is This For?

This playbook is written for relief coordinators, district disaster officers, and NGO operations managers who need to make real decisions about where to send supplies before a flood hits. You don't need a technical background to use it.

---

### The Core Idea

Instead of waiting for floodwaters to rise and then scrambling to truck in supplies, this system looks at weather patterns, historical flood data, and district-level demographics to tell you what each district will probably need over the next 7 to 30 days. That gives you a window to pre-position resources at the right warehouses before roads get cut off.

---

### How to Use the Dashboard (Step by Step)

**Step 1: Check the Overview page every morning.**
Open the dashboard and look at the Overview tab. It shows you all 38 Bihar districts ranked by risk score. Pay attention to anything marked "Critical" or "High" — those are the ones that need your attention today.

**Step 2: Look at the Risk Map.**
The map gives you a visual sense of where trouble is brewing. Red and orange circles mean elevated risk. Click on any circle to see the district name and its current risk score. This helps you see clusters — if three neighbouring districts are all turning red, that suggests a regional event is building.

**Step 3: Run a forecast for specific districts.**
Go to the Forecasting page. Pick the district you're concerned about, set the forecast horizon (we recommend 7 days for tactical planning, 14 days for logistics planning, 30 days for budget planning), and plug in the weather forecast from IMD. Hit "Generate Forecast."

The system will tell you how many food kits, medical kits, ORS packets, litres of drinking water, and tarpaulins that district is likely to need. It also shows confidence intervals — the actual demand will probably fall somewhere between the lower and upper bounds.

**Step 4: Read the recommendations.**
The Recommendations page generates plain-English summaries for every at-risk district. Something like:

> "Darbhanga district has critical flood risk. Immediate action required. Pre-position 6,200 food kits, 2,800 medical kits, and 15,000 ORS packets within the next 7 days. Ensure 42,000 litres of drinking water are available at distribution centres. Deploy 1,800 tarpaulins to temporary shelters."

You can download these as a CSV and share them with your logistics team or email them to the district collector's office.

**Step 5: Act on the numbers.**
Once you have the predictions:
- Check your current warehouse stock against the predicted need.
- If there's a gap, trigger procurement or request inter-district transfers.
- Alert your field teams in the at-risk districts.
- Coordinate with the district administration on shelter readiness.

---

### When to Run Forecasts

| Situation | Horizon | How Often |
|-----------|---------|-----------|
| Normal monsoon monitoring | 14 days | Twice a week |
| IMD issues heavy rainfall warning | 7 days | Daily |
| Active flooding in neighbouring districts | 7 days | Daily |
| Budget planning for monsoon season | 30 days | Once at start of June |
| Post-event restocking | 7 days | As needed |

---

### Understanding the Risk Scores

The system assigns every district a risk score from 0 to 100 based on four factors:

- **Rainfall intensity** (35% weight) — How much rain is falling compared to historical extremes.
- **Flood severity** (30% weight) — Current or expected flood level on a 0-5 scale.
- **River water level** (25% weight) — Gauge readings relative to historical peaks.
- **Monsoon season flag** (10% weight) — Whether we're currently in June-September.

| Score | Level | What It Means |
|-------|-------|---------------|
| 0-30 | Low | Maintain normal readiness. No special action needed. |
| 30-50 | Moderate | Increase monitoring frequency. Check stock levels. |
| 50-70 | High | Pre-position supplies. Alert field teams. Brief local authorities. |
| 70-100 | Critical | Immediate mobilisation. Deploy advance teams. Open shelters. |

---

### Per-Capita Supply Norms (SPHERE Standards)

When the model predicts a number of kits, here's what each one covers:

| Resource | Covers | Standard |
|----------|--------|----------|
| Food Kit | 1 family (5 people) for 3 days | Dry rations: rice, dal, oil, salt, sugar |
| Medical Kit | 10 people for 1 week | Basic first aid, ORS, paracetamol, bandages |
| ORS Packet | 1 person for 1 day | Oral Rehydration Salts, critical for preventing dehydration |
| Drinking Water | 1 person for 1 day (min 2.5L) | Clean potable water |
| Tarpaulin | 1 family shelter | 4m x 5m waterproof sheet |

---

### What the Model Cannot Do

Let's be honest about the limits:

1. **It doesn't predict where the water will go.** The model forecasts demand based on historical patterns and weather data. It doesn't do hydrological modelling — it won't tell you which specific villages will flood.

2. **It assumes the future looks roughly like the past.** If there's a dam breach or an unprecedented weather pattern, the model's predictions will be less reliable. Use your judgement.

3. **It doesn't account for supply chain disruptions.** The model tells you what's needed, not whether roads will be passable for delivery trucks.

4. **MAPE numbers look high on paper.** The Mean Absolute Percentage Error is inflated because many days have near-zero demand. During actual flood events — when the predictions matter most — the model is considerably more accurate.

---

### Recommended Workflow for an NGO

```
Weekly (June-September):
  Monday   - Run 14-day forecasts for all high-risk districts
  Tuesday  - Compare predictions against warehouse inventory
  Wednesday - Place procurement orders for any shortfalls
  Thursday - Coordinate with district authorities on logistics
  Friday   - Update field teams with latest risk assessments

During Active Events:
  Daily    - Run 7-day forecasts for affected districts
  Daily    - Share recommendations CSV with all stakeholders
  Daily    - Track actual distribution vs. predicted needs
  Weekly   - Review model accuracy and adjust if needed
```

---

### Getting Help

- **Dashboard issues**: Check that both the FastAPI backend (port 8000) and Streamlit (port 8501) are running.
- **Unexpected predictions**: Look at the input weather data — garbage in, garbage out. Cross-check with IMD's latest bulletin.
- **Model retraining**: After each monsoon season, feed the actual distribution data back into the system and retrain. The model gets better with more real-world data.

---

*This playbook was prepared as part of the AI-Powered Disaster Resource Allocation Forecasting System project for Bihar Disaster Management operations.*
