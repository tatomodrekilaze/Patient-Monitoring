# Validation

Every feature was run against public data with `tools/eval_*.py`. Datasets aren't included
(licenses), get them from the original sources. Small samples, some thresholds tuned on the same
data. Not a clinical study.

| feature | data | result | caveat |
|---|---|---|---|
| resp rate | MPSC-RR | 1.2-1.9 /min vs reference, 0 false apnea | chest must be visible |
| heart rate | UBFC-rPPG vs oximeter | MAE 1.7 bpm, r 0.95 | still, lab light |
| falls | UR Fall, 70 clips | 23/30 falls, 2/40 false | 2 s after impact |
| posture | UR Fall, 9825 frames | 77.4% lying agreement | |
| bed exit | UR Fall adl-36/37 reversed | ok | 2 clips |
| hand signs | HaGRID, 3600 images | 87.7%, fist -> false request 5% | photos |
| eyes closed | 600 faces | 95.3% closed, 100% open | |
| face test | palsynet | 18/23 palsy, 0/22 healthy | tuned on same set |
| chair stand | HMDB51 | 0/32 false stands | |
| EVM view | synthetic | x12.9 motion, x36 colour | display only |

Run: `python -I tools/eval_urfall.py <dir>` (same for hagrid, rppg, face, chair; `eval_breathing_video.py <video>`).

## Removed

Built, then dropped because there was no data or results were bad: cardiac arrest / CPR, sepsis,
NEWS2 / deterioration score, sleep apnea index, seizure / agitation / delirium, tremor and other
simulated-only exams, pain face, passive face droop, gait speed, fall risk score, fall impact speed.

## Bugs worth remembering

- Fixed-size OpenVINO export letterboxes differently, +2 false falls. Use the dynamic export.
- Body cut off by the frame: yolo invents hips at the edge, chest box lands on the face, fake apnea.
- Dead websocket clients never closed; after ~20 h the AI was at 3 fps.
- Wi-Fi cameras deliver in bursts, timestamps come from stream pts, not arrival.
