"""Prepare, do not install while initial campaign owns the official binary."""
import json
import runtime as r

source=r.REPO/'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args_benchmark.cpp'
s=source.read_text()
def edit(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
edit('    int samples             = 30;', '    int samples             = 30;\n    int launches_per_sample = 1; // Explicit diagnostic; formal default stays one.')
edit('        else if(name == "correctness-repeats")',
     '        else if(name == "launches-per-sample")\n            options.launches_per_sample = parse_positive_or_zero(name, value);\n        else if(name == "correctness-repeats")')
edit('''if((options.time && options.samples <= 0) || options.correctness_repeats <= 0)
        throw std::runtime_error("计时开启时 samples、所有模式下 correctness-repeats 必须为正数");''',
     '''if((options.time && options.samples <= 0) || options.correctness_repeats <= 0)
        throw std::runtime_error("计时开启时 samples、所有模式下 correctness-repeats 必须为正数");
    if(options.launches_per_sample <= 0 || options.launches_per_sample > 256)
        throw std::runtime_error("launches-per-sample 必须在 1..256 范围内");''')
edit('float measure(hipStream_t stream, const std::function<void()>& launch)',
     'float measure(hipStream_t stream, const std::function<void()>& launch, int launches)')
edit('''        hip_check(hipEventRecord(start_, stream), "hipEventRecord(start)");
        launch();
        hip_check(hipEventRecord(stop_, stream), "hipEventRecord(stop)");''',
     '''        hip_check(hipEventRecord(start_, stream), "hipEventRecord(start)");
        if(launches == 1)
            launch();
        else
            for(int i = 0; i < launches; ++i)
                launch();
        hip_check(hipEventRecord(stop_, stream), "hipEventRecord(stop)");''')
edit('        return milliseconds;', '        return milliseconds / static_cast<float>(launches);')
assert s.count('timer.measure(stream, launch_ck)')==2
s=s.replace('timer.measure(stream, launch_ck)','timer.measure(stream, launch_ck, options.launches_per_sample)')
edit('timer.measure(stream, launch_blas)','timer.measure(stream, launch_blas, options.launches_per_sample)')
edit('               << " include_args_build=" << options.include_args_build',
     '               << " launches_per_sample=" << options.launches_per_sample\n               << " include_args_build=" << options.include_args_build')
edit('       << "  --num-cu=0 --validate=0|1 --include-args-build=0|1\\n"',
     '       << "  --launches-per-sample=1 (1..256; diagnostic; timings are per launch)\\n"\n       << "  --num-cu=0 --validate=0|1 --include-args-build=0|1\\n"')
target=r.ROOT/'benchmark_launch_diagnostic_pending.cpp';assert not target.exists();target.write_text(s)
(r.ROOT/'benchmark_launch_diagnostic_pending.json').write_text(json.dumps(dict(at=r.stamp(),source=str(source),source_sha256=r.sha(source),prepared_source_sha256=r.sha(target),installed=False,
    reason='Native76 same-card single .023840 vsrepeat20 .017092ms; separate launch-boundary diagnostic without changing formal one-launch/frozen-R protocol',
    scope='76/620/621, existing current/control and newK32 routes, launches1vs20; no new pipeline or attribute expansion'),indent=2))
print('LAUNCH_DIAGNOSTIC_PREPARED_NOT_INSTALLED')
