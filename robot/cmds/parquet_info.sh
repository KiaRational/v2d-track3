python - <<'PY'
import pyarrow.parquet as pq, numpy as np, glob
f = glob.glob("/scratch/kghasemz/v2d/robot/hmd/ego_recon/processed/sequence_id=ep39_dustpan/robot_name=sharpa_wave/*.parquet")[0]
t = pq.read_table(f)
print(t.num_rows, "rows")
for n in t.column_names:
    c = t.column(n)
    try:
        v = np.array(c.to_pylist()[0], dtype=object)
        s = np.asarray(c.to_pylist()[0], dtype=float)
        print(f"{n:50s} {s.shape} first={np.round(s.ravel()[:8], 3).tolist()}")
    except Exception as e:
        print(f"{n:50s} {str(c.type)[:60]}  {str(c.to_pylist()[0])[:120]}")
PY
