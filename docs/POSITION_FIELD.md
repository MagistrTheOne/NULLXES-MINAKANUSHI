# NullxesPositionField

Native position is multidimensional:

```text
P_i = (sequence, physical_time, space, episode, memory_age, source)
```

Physical time is not token index. Observation time, processing time, event
time, and prediction horizon are different quantities. NPF encodes the first
and memory age; runtime telemetry records processing latency separately.

Independent encoders:

| Axis | Module | Input |
|---|---|---|
| sequence | `position/sequence.py` | stream order |
| time | `position/temporal.py` | timestamp (seconds) |
| space | `position/spatial.py` | (x,y,z) gated by validity |
| episode | `position/episode.py` | episode index |
| memory age | `position/memory_age.py` | log1p(t_now - t_i) |
| source | `position/source.py` | stream id embedding |

A trainable mixer fuses the six embeddings. Missing spatial coordinates
produce a zero spatial embedding; they are not invented.

This is not RoPE and is not aliased to another positional system.
