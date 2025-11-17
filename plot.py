# plot.py

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Cấu hình ---
RESULTS_DIR = "results"
# Chỉ so sánh hai thuật toán cốt lõi đã được chạy đồng bộ
ALGORITHMS = ["mappo", "qmappo"] 
PLOT_LABELS = {
    "mappo": "MAPPO",
    "qmappo": "Q-MAPPO (Ours)"
}
PALETTE = {
    "MAPPO": "#1f77b4",      # Màu xanh dương chuẩn của Matplotlib
    "Q-MAPPO (Ours)": "#ff7f0e" # Màu cam chuẩn của Matplotlib
}

# Cấu hình chung cho Matplotlib để có chất lượng bài báo
plt.style.use('seaborn-v0_8-whitegrid')
FONT_SIZE = 14
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2, # Tiêu đề to hơn một chút
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'figure.figsize': (10, 6), # Kích thước hình chữ nhật
    'lines.linewidth': 2.5
})

def load_and_process_data(algorithms: list[str], results_dir: str):
    """
    Tải tất cả các file CSV cho các thuật toán được chỉ định, gộp chúng lại.
    """
    all_dfs = []
    for algo in algorithms:
        algo_files = [f for f in os.listdir(results_dir) if f.startswith(algo) and f.endswith('.csv')]
        
        if not algo_files:
            print(f"Warning: No CSV files found for algorithm '{algo}' in '{results_dir}'")
            continue
            
        print(f"Loading data for {algo}: {algo_files}")
        
        for i, filename in enumerate(algo_files):
            df = pd.read_csv(os.path.join(results_dir, filename))
            df['seed'] = i + 1
            df['algorithm'] = PLOT_LABELS[algo]
            all_dfs.append(df)
            
    if not all_dfs:
        raise ValueError("No data loaded. Check RESULTS_DIR and ALGORITHMS.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.dropna(subset=['mean_episodic_reward'], inplace=True)

    return full_df

def plot_learning_curves(data: pd.DataFrame):
    """
    Vẽ biểu đồ đường cong học với trung bình và khoảng tin cậy 95%.
    """
    print("\n--- Generating Learning Curve Plot ---")
    
    fig, ax = plt.subplots() # Tạo figure và axes để kiểm soát tốt hơn
    
    sns.lineplot(
        data=data,
        x="global_step",
        y="mean_episodic_reward",
        hue="algorithm",
        palette=PALETTE,
        errorbar=("ci", 95), # Khoảng tin cậy 95%
        ax=ax
    )
    
    ax.set_title("Performance Comparison on Satellite Task Scheduling")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episodic Reward")
    
    # Định dạng lại trục x cho dễ đọc (ví dụ: 1e6 thay vì 1000000)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Lấy legend handle và chỉnh sửa
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Algorithm")
    
    fig.tight_layout()
    
    output_filename = os.path.join(RESULTS_DIR, "learning_curve_comparison.pdf")
    fig.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"Learning curve saved to {output_filename}")
    plt.show()

def generate_performance_table(data: pd.DataFrame):
    """
    Tính toán và in ra bảng hiệu suất cuối cùng, dựa trên 10% cuối của quá trình huấn luyện.
    """
    print("\n--- Generating Final Performance Table ---")
    
    # Lấy ngưỡng 90% timesteps cho mỗi thuật toán
    last_10_percent_step = data.groupby('algorithm')['global_step'].max() * 0.9
    
    final_performance_data = pd.DataFrame()
    for algo in data['algorithm'].unique():
        threshold = last_10_percent_step[algo]
        subset = data[(data['algorithm'] == algo) & (data['global_step'] >= threshold)]
        final_performance_data = pd.concat([final_performance_data, subset])

    # Tính trung bình và độ lệch chuẩn của reward cuối cùng cho mỗi thuật toán qua các seed
    summary = final_performance_data.groupby('algorithm')['mean_episodic_reward'].agg(['mean', 'std']).reset_index()
    
    print("Final Performance (averaged over last 10% of training):")
    print("=" * 65)
    print(f"{'Algorithm':<25} | {'Mean Reward ± Std Dev'}")
    print("-" * 65)
    
    # Sắp xếp lại thứ tự để trình bày logic hơn
    summary['algorithm'] = pd.Categorical(summary['algorithm'], categories=[PLOT_LABELS[a] for a in ALGORITHMS], ordered=True)
    summary = summary.sort_values('algorithm')

    for _, row in summary.iterrows():
        # Định dạng output cho đẹp, phù hợp để copy vào paper
        print(f"{row['algorithm']:<25} | {row['mean']:.2f} ± {row['std']:.2f}")
        
    print("=" * 65)

    # Lưu bảng dưới dạng CSV để dễ dàng xử lý sau này
    table_filename = os.path.join(RESULTS_DIR, "performance_table_comparison.csv")
    summary.to_csv(table_filename, index=False)
    print(f"Performance table saved to {table_filename}")


if __name__ == "__main__":
    try:
        # Tải và xử lý dữ liệu
        full_data = load_and_process_data(ALGORITHMS, RESULTS_DIR)
        
        # Vẽ biểu đồ đường cong học
        plot_learning_curves(full_data)
        
        # Tạo bảng hiệu suất
        generate_performance_table(full_data)
        
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find result files in the '{RESULTS_DIR}' directory. Make sure the experiments have finished and CSV files are present.")
