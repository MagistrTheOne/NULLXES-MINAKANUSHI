# YMBOT-C — MINAKANUSHI laboratory kit

Sealed partner folder. **Not the NULLXES research repository.**

```text
architecture : MINAKANUSHI
organization : NULLXES
envelope     : YMBOT-C
accepted     : NO
pwm          : false
```

| File | What it is |
|---|---|
| `mina_loop.py` | L0 closed loop (copy this one file if you must) |
| `run_lab.py` | `l0` / `l1` / `l2` |
| `LAB_GUIDE.en.md` | English bring-up + GPU matrix |
| `LAB_GUIDE.zh.md` | 中文实验指南 |
| `LICENSE_EVALUATION.md` | partner lab license |

```text
python -m pip install -r requirements.txt
python run_lab.py l0 --selftest
python run_lab.py l0 --steps 24 --report artifacts/l0.json
python run_lab.py l1 --device cpu --steps 40
python -m pytest tests -q
```

L2 (6.8B infer) needs the NULLXES partner wheel, an 80 GB-class GPU, and a Hub `*.mina`. This folder will **refuse** to build 6.8B on a laptop.
