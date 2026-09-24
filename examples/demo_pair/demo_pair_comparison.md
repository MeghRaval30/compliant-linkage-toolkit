# demo_pair: the same four-bar, built two ways

One linkage. Same link lengths, same coupler point, same base footprint and the same M3 hole pattern, so one fixture position serves both parts and both draw through a pen hole at the same coordinate.

> **NOT A PHYSICAL PREDICTION - placeholder inputs used: kobra2_neo.design_rules.min_flexure_thickness_mm, PLA.properties.allowable_strain, PLA.properties.youngs_modulus_MPa**
>
> Every row below that depends on material data is a prediction from placeholder constants. It shows the pipeline runs; it is not what the printed parts will do. Waiting on: `kobra2_neo.design_rules.min_flexure_thickness_mm`, `PLA.properties.allowable_strain`, `PLA.properties.youngs_modulus_MPa`

| | pin-jointed (rigid) | compliant (flexures) |
|---|---|---|
| **printed bodies** | 1 | 1 |
| **fasteners** | 0 | 0 |
| **assembly** | free four joints by hand | none |
| **moving joints** | 4 sliding pin joints | 4 flexures |
| **backlash** | 0.35 mm radial pin clearance | none |
| **friction** | sliding, at every pin | none; energy stored elastically |
| **peak input torque (N&middot;mm)** | not predicted | 47.0 (PRBM) / 30.9 (FEA) |
| **coupler path vs the ideal rigid path** | up to &plusmn;0.35 mm from pin clearance alone | 0.321 mm mean, 0.628 mm max (FEA) |
| **flexure strain margin** | n/a -- no flexures | 1.25x at joint A (0.799% of 1.00%) against a PLACEHOLDER allowable |
| **range of motion** | continuous rotation, if the links clear | 22.0 deg of input, and no more |
| **envelope (mm)** | 60 x 65 x 11 | 60 x 86 x 7 |
| **ESTIMATED mass (g)** | 10.5 | 24.8 |
| **ESTIMATED print (min)** | 23 | 56 |

## Notes

- **printed bodies.** The compliant part is one piece by definition. The rigid part is one piece only because the pins print already assembled; the bolt variant is four.
- **backlash.** The compliant joint has no clearance to take up: it is solid material. The pin clearance is a design choice that has not been printed yet, so whether it frees off at all is still open.
- **friction.** Which is why the compliant part needs torque to *hold* a position and the rigid one does not.
- **peak input torque (N&middot;mm).** The path is fixed by geometry under a prescribed input, so only torque can discriminate between the two stiffness models -- which is why they disagree here by about half and agree exactly on the path.
- **coupler path vs the ideal rigid path.** The rigid figure is a bound from the clearance, not a prediction: a pin can sit anywhere in its hole. The compliant figure is a solved result. They are the same order, which is the point -- neither construction reproduces the ideal path, and only one of them can be predicted before printing.
- **flexure strain margin.** The number that decides whether the compliant part survives. It has no rigid counterpart: a pin joint does not care how far it turns, which is exactly the trade being made.
- **range of motion.** The hard limit on the compliant side, and the honest cost of the technology: a flexure cannot rotate continuously, so there is no full-revolution option anywhere in this toolkit.
- **ESTIMATED print (min).** Solid volume at an assumed deposition rate. A planning figure, not a measurement; the slicer's number supersedes it.

## Deliberately blank

These are not oversights. Filling them in would claim the comparison the demo exists to make.

- **The rigid part's input torque.** There is no friction model here and no measurement of the printed pins, so it is not predicted. Whatever it takes to drive the pin-jointed part is friction, and friction is the thing the compliant design removes -- claiming a number for it would be claiming the result.
- **Both parts' measured coupler paths.** Nothing has been printed. The pen holes exist so the two curves can be drawn on one sheet of paper and compared directly; until then the measured series is absent from every figure, not zero.
- **Whether the print-in-place pins free off at all.** The clearance is a design choice carried over from common practice. The first print is the experiment.

---

cmtool 0.1.0.dev0 @ `0b26668436e5` &middot; config `ecf9b14c9554` &middot; 2026-09-24T13:36:08+00:00
