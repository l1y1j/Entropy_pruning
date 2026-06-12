import os
import re
import sys
import argparse
import matplotlib.pyplot as plt


def parse_log_file(log_path, param_name):
    """从日志文件中提取指定参数的值"""
    values = []
    with open(log_path, 'r') as f:
        for line in f:
            # 匹配形如 "mAP_50: 0.123" 或 "mAP_50 = 0.123" 或 "mAP_50 0.123" 的模式
            pattern = rf'{re.escape(param_name)}\s*[:=]?\s*([-+]?\d*\.?\d+)'
            match = re.search(pattern, line)
            if match:
                values.append(float(match.group(1)))
    return values


def plot_param_curve(values, param_name, output_path=None):
    """绘制参数折线图"""
    if not values:
        print(f"未找到参数 '{param_name}' 的数据")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(values, marker='o', markersize=4, linewidth=1.5)
    plt.xlabel('Index')
    plt.ylabel(param_name)
    plt.title(f'{param_name} over iterations')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"图片已保存至: {output_path}")
    else:
        plt.show()


# ============ 配置区 ============
LOG_FILE = "/data/linyujie/projects/Entropy_pruning/outputs/v3_kl/2/20260530_130328/20260530_130328.log"
PARAM_NAME = "mAP_50"
OUTPUT_PATH = "/data/linyujie/projects/Entropy_pruning/outputs/v3_kl/2/20260530_130328/mAP_50.png"  # 设为 None 则显示窗口，设为路径则保存图片
# ============ 配置区 ============


def main():
    # 命令行模式：如果传入参数则使用命令行，否则使用配置区
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='从日志文件中绘制参数折线图')
        parser.add_argument('log_file', help='日志文件路径')
        parser.add_argument('param', help='要绘制的参数名, 如 mAP_50')
        parser.add_argument('-o', '--output', help='输出图片路径(可选)', default=None)
        args = parser.parse_args()
        log_file = args.log_file
        param_name = args.param
        output_path = args.output
    else:
        log_file = LOG_FILE
        param_name = PARAM_NAME
        output_path = OUTPUT_PATH

    if not os.path.exists(log_file):
        print(f"错误: 文件 '{log_file}' 不存在")
        return

    values = parse_log_file(log_file, param_name)
    print(f"找到 {len(values)} 个数据点")
    plot_param_curve(values, param_name, output_path)


if __name__ == '__main__':
    main()