import pandas as pd
import matplotlib.pyplot as plt

# Load results. To compare CC6.1 vs CC8.9, load both CSVs and merge:
#   df61 = pd.read_csv('bench_deserno_lmp_kokkos_61_YYYYMMDD_HHMM.csv')
#   df89 = pd.read_csv('bench_deserno_lmp_kokkos_89_YYYYMMDD_HHMM.csv')
#   df = pd.concat([df61, df89])

machine = "md43"
machine = "md69"
machine = "md69_device"
machine = "md69_device_uvm_off"

df = pd.read_csv(f'results_{machine}.csv')


# Normalise GPU column to bool
if df['GPU'].dtype == object:
    df['GPU'] = df['GPU'].astype(str).str.strip().str.lower().map({'true': True, 'false': False})


#########################
##### TIMESTEPS/SECS ####
#########################

plt.figure(figsize=(10, 6))

gpu_enabled = df[df['GPU'] == True]
for binary in sorted(gpu_enabled['LMP_Binary'].unique()):
    for omp_val in sorted(gpu_enabled['OMP'].unique()):
        subset = (gpu_enabled[(gpu_enabled['LMP_Binary'] == binary) &
                              (gpu_enabled['OMP'] == omp_val)]
                  .sort_values('N'))
        if not subset.empty:
            plt.plot(subset['N'], subset['TPS'], marker='o',
                     label=f'GPU ({binary}, OMP={omp_val})')

gpu_disabled = df[df['GPU'] == False]
for mpi_val in sorted(gpu_disabled['MPI'].unique()):
    subset = gpu_disabled[gpu_disabled['MPI'] == mpi_val].sort_values('N')
    if not subset.empty:
        plt.plot(subset['N'], subset['TPS'], marker='x', linestyle='--',
                 label=f'MPI={mpi_val} (CPU)')

plt.title('Deserno bilayer: Throughput vs. System Size')
plt.xlabel('System Size (N atoms)')
plt.ylabel('Steps/s')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(title="Configuration")
plt.tight_layout()
plt.savefig(f"deserno_steps_{machine}.png", dpi=200, transparent=True)
plt.show()
plt.close()

#########################
######   SPEEDUP   ######
#########################

# Baseline: CPU, MPI=1, OMP=1 (first OMP value present)
baseline_omp = sorted(gpu_disabled['OMP'].unique())[0]
baseline = (
    gpu_disabled[(gpu_disabled['MPI'] == 1) & (gpu_disabled['OMP'] == baseline_omp)]
    .set_index('N')['TPS']
)

if baseline.empty:
    print("No MPI=1 CPU baseline found; skipping speedup plot.")
else:
    plt.figure(figsize=(10, 6))

    # GPU speedup lines — one per (binary, OMP thread count)
    for binary in sorted(gpu_enabled['LMP_Binary'].unique()):
        for omp_val in sorted(gpu_enabled['OMP'].unique()):
            subset = (gpu_enabled[(gpu_enabled['LMP_Binary'] == binary) &
                                  (gpu_enabled['OMP'] == omp_val)]
                      .sort_values('N'))
            speedup = subset.set_index('N')['TPS'].div(baseline).dropna()
            if not speedup.empty:
                plt.plot(speedup.index, speedup.values, marker='o',
                         label=f'GPU ({binary}, OMP={omp_val})')

    # CPU MPI speedup lines
    for mpi_val in sorted(gpu_disabled['MPI'].unique()):
        subset = (gpu_disabled[(gpu_disabled['MPI'] == mpi_val) &
                               (gpu_disabled['OMP'] == baseline_omp)]
                  .sort_values('N'))
        speedup = subset.set_index('N')['TPS'].div(baseline).dropna()
        if not speedup.empty:
            plt.plot(speedup.index, speedup.values, marker='x', linestyle='--',
                     label=f'MPI={mpi_val} (CPU)')

    plt.title('Deserno bilayer: Speedup vs. System Size (baseline: MPI=1 CPU)')
    plt.xlabel('System Size (N atoms)')
    plt.ylabel('Speedup')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(title="Configuration")
    plt.tight_layout()
    plt.savefig(f"deserno_speedup_{machine}.png", dpi=200, transparent=True)
    plt.show()
    plt.close()
