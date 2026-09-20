# data/

Bulk research data lives here and is **not committed to git**: STL/STEP exports, video,
tracked paths, generated samples and the Parquet index. Releases go to Zenodo with a DOI
and a `DATASET_CARD.md` (Phase C), licensed CC BY 4.0 (`../LICENSE-DATA`).

Git LFS is deliberately not used: quota limits make it painful to undo on a student account.

Layout (Phase B onward):

```
data/
  samples/<sample_id>/
    sample.json          # validated against src/cmtool/schema/json/sample.schema.json
    paths/rigid.csv prbm.csv fea_beam.csv measured_*.csv
    cad/mechanism.step mechanism.stl
  index.parquet          # one row per sample, for fast filtering
```
