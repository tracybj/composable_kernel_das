# Measurement and launch facts checked from current code

Formal CK uses `EventTimer::measure`: record start event, call the public v1
run API/default selector once, record stop event, synchronize stop. `samples`
and its `repeat` alias count these independent single-launch measurements.
`include-args-build=0` omits device argument construction from each timed call;
the public run API still validates the problem and executes the selector.
Profiler dispatch time and formal event time therefore have different scopes.
Do not infer a GPU-pipeline gain from a counter change alone.

The benchmark leaves `Options::num_cu=0` unless explicitly supplied. This
campaign does not supply it. V7/small-M new launch code chooses slots=1 in that
case, so its grid is the exact tile count. A hypothesis about rounding low-CTA
grids to80 does not apply to these formal commands. Actual dispatch records
remain authoritative, especially for the separate persistent4003 fallback.

`detect_arch` calls `get_hcu_target_enum`; device properties are initialized
once in `hcu_device_ctx`. It does not query device properties from HIP on each
timed call. String/map/environment processing remains CPU work, but its cost
has not been measured or shown to be the bottleneck here.

Raw K-stage width is `G::kStageK` inside the V7 pipeline. Inherited framework
shape names may encode a different K block dimension; do not use that name
alone to infer16/32-stage memory requests. Raw `RunGemmV7Uniform` passes full K.

Possible scheduling or host-overhead diagnostics require representative timing
and profile evidence first. No such additional optimization is implemented or
claimed in these notes. Any multiple-launch diagnostic would need an explicit
new default-off option and separate results, while formal acceptance stays
single-launch through the public API/default selector.
