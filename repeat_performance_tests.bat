FOR /L %i IN (1,1,10) DO (
    python performance_correlate.py runs\run_performance_correlate.txt
)

FOR /L %i IN (1,1,10) DO (
    python performance_split_nmf.py runs\run_performance_split.txt
)
